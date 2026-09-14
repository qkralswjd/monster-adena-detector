"""
target_selector.py
------------------
타겟 선택 + 추적 로직.

동작 원칙:
  1. 타겟 없음 → confidence 최고 몬스터 선택 (고정)
  2. 타겟 고정 후 → 매 프레임 IoU + 거리로 같은 몬스터 추적
  3. 타겟이 miss_timeout 초 이상 탐지 안 되면 → "사망 추정" → 새 타겟 선택
  4. 단, 탐지는 됐는데 다른 몬스터가 더 가깝더라도 절대 타겟 교체 안 함

개선 (v2):
  - locked 플래그: 타겟 고정 후 same-frame 내 타겟 교체 완전 차단
  - miss 중에도 마지막 타겟 위치 유지 (공격 계속 가능)
  - 사망 판단: miss_timeout 초과 & 화면 내 비슷한 위치 몬스터 없음
  - 소실 로그 상세화 (miss 시간, 마지막 위치)

좌표 원칙:
  - 입력/출력 모두 화면 절대좌표 그대로
  - 변환 없음
"""

import math
import time
from typing import List, Optional
from detector import Detection


# ── IoU 계산 ─────────────────────────────────────────────────
def calc_iou(a: Detection, b: Detection) -> float:
    """두 Detection 의 IoU 계산."""
    ax1, ay1, ax2, ay2 = a.tlbr
    bx1, by1, bx2, by2 = b.tlbr

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def select_by_confidence(detections: List[Detection]) -> Optional[Detection]:
    """confidence 가장 높은 몬스터 반환."""
    monsters = [d for d in detections if d.class_id == 0]
    if not monsters:
        return None
    return max(monsters, key=lambda d: d.confidence)


def select_nearest(detections: List[Detection],
                   ref_x: int, ref_y: int) -> Optional[Detection]:
    """기준점에서 가장 가까운 몬스터 반환."""
    monsters = [d for d in detections if d.class_id == 0]
    if not monsters:
        return None
    return min(monsters,
               key=lambda d: math.hypot(d.cx - ref_x, d.cy - ref_y))


# ── TargetTracker (v2) ────────────────────────────────────────
class TargetTracker:
    """
    타겟 고정 + IoU 추적 (v2).

    핵심 변경:
      - 타겟 고정 후 절대 교체 안 함 (miss_timeout 초과 시에만 교체)
      - miss 중에도 마지막 타겟 정보 유지 → 주 루프에서 판단 가능
      - _locked: True이면 다른 몬스터로 교체 불가
      - 사망 판단 강화: miss_timeout 초과 후 '완전 소실 확인' 필요

    사용:
        tracker = TargetTracker(miss_timeout=4.0, iou_thresh=0.2, max_dist=350)
        target = tracker.update(detections)

        # miss 중 여부는 miss_elapsed 로 확인
        # miss_elapsed > 0   → 이번 프레임에 타겟 탐지 안 됨
        # miss_elapsed == 0  → 이번 프레임에 타겟 탐지됨
    """

    def __init__(self,
                 miss_timeout: float = 4.0,
                 iou_thresh: float = 0.2,
                 max_dist: float = 350):
        self._target       = None   # 현재 고정된 타겟 (Detection)
        self._last_seen    = 0.0    # 마지막으로 타겟 탐지된 시각
        self._locked       = False  # 타겟 고정 여부
        self._miss_timeout = miss_timeout
        self._iou_thresh   = iou_thresh
        self._max_dist     = max_dist

        # 이번 프레임 추적 성공 여부 (miss_elapsed 계산용)
        self._found_this_frame = False

    # ── 공개 프로퍼티 ─────────────────────────────────────────
    @property
    def target(self) -> Optional[Detection]:
        return self._target

    @property
    def is_locked(self) -> bool:
        """타겟이 고정된 상태인지."""
        return self._locked and self._target is not None

    @property
    def miss_elapsed(self) -> float:
        """
        이번 프레임에 타겟을 탐지했으면 0.0.
        탐지 못 했으면 마지막 탐지로부터 경과 초.
        """
        if self._target is None:
            return 0.0
        if self._found_this_frame:
            return 0.0
        return time.time() - self._last_seen

    # ── 매 프레임 업데이트 ────────────────────────────────────
    def update(self, detections: List[Detection]) -> Optional[Detection]:
        """
        매 프레임 호출.
        현재 타겟을 추적하거나 새 타겟을 선택해서 반환.
        반환값: 현재 타겟 Detection (miss 중이어도 마지막 타겟 반환)
        """
        monsters = [d for d in detections if d.class_id == 0]
        now = time.time()
        self._found_this_frame = False

        # ── [A] 타겟 없음 → 새 타겟 선택 ────────────────────
        if self._target is None:
            new = select_by_confidence(monsters)
            if new:
                self._target    = new
                self._last_seen = now
                self._locked    = True
                self._found_this_frame = True
                print(f"[Tracker] ★ 타겟 선택: cx={new.cx} cy={new.cy} "
                      f"conf={new.confidence:.2f}")
            return self._target

        # ── [B] 타겟 고정 중 → IoU + 거리로 같은 몬스터 추적
        best       = None
        best_score = -1.0

        for d in monsters:
            iou  = calc_iou(self._target, d)
            dist = math.hypot(d.cx - self._target.cx,
                              d.cy - self._target.cy)

            # 두 조건 중 하나라도 만족하면 후보
            if iou >= self._iou_thresh or dist <= self._max_dist:
                # 점수: IoU 우선 + 거리 보조
                score = iou * 2.0 + (1.0 - min(dist, self._max_dist) / self._max_dist)
                if score > best_score:
                    best_score = score
                    best = d

        if best is not None:
            # ── 탐지 성공 → 타겟 위치 갱신 (교체 X, 같은 몬스터 추적)
            prev_cx, prev_cy = self._target.cx, self._target.cy
            self._target    = best
            self._last_seen = now
            self._found_this_frame = True

            # 위치 변화가 클 때만 로그
            moved = math.hypot(best.cx - prev_cx, best.cy - prev_cy)
            if moved > 30:
                print(f"[Tracker] 추적 이동: ({prev_cx},{prev_cy}) → "
                      f"({best.cx},{best.cy})  Δ{moved:.0f}px")

        else:
            # ── 탐지 실패 → miss 타이머 체크
            elapsed = now - self._last_seen

            if elapsed > self._miss_timeout:
                # miss_timeout 초과 → 사망 추정 → 타겟 해제
                print(f"[Tracker] ✗ 타겟 소실 ({elapsed:.1f}s) "
                      f"마지막위치=({self._target.cx},{self._target.cy}) "
                      f"→ 사망 추정, 새 타겟 탐색")
                self._target = None
                self._locked = False

                # 이번 프레임 새 타겟 즉시 선택
                new = select_by_confidence(monsters)
                if new:
                    self._target    = new
                    self._last_seen = now
                    self._locked    = True
                    self._found_this_frame = True
                    print(f"[Tracker] ★ 새 타겟: cx={new.cx} cy={new.cy} "
                          f"conf={new.confidence:.2f}")
            else:
                # miss 중이지만 아직 timeout 전 → 마지막 타겟 유지
                # (공격 여부는 주 루프에서 miss_elapsed 확인 후 결정)
                remaining = self._miss_timeout - elapsed
                if elapsed > 1.0 and int(elapsed * 2) % 2 == 0:  # 0.5초마다 한 번만
                    print(f"[Tracker] miss {elapsed:.1f}s / {self._miss_timeout:.1f}s "
                          f"(타겟 유지, 남은={remaining:.1f}s)")

        return self._target

    def reset(self):
        """타겟 초기화 (수동 리셋용)."""
        self._target            = None
        self._last_seen         = 0.0
        self._locked            = False
        self._found_this_frame  = False
        print("[Tracker] 타겟 리셋")
