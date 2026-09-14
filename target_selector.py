"""
target_selector.py
------------------
타겟 선택 + 추적 로직.

동작:
  1. 타겟 없음 → confidence 최고 몬스터 선택 (고정)
  2. 타겟 고정 후 → 매 프레임 IoU + 거리로 같은 몬스터 추적
  3. 타겟이 miss_timeout 초 이상 탐지 안 되면 → 새 타겟 선택

좌표 원칙:
  - 입력/출력 모두 화면 절대좌표 그대로
  - 변환 없음
"""

import math
import time
from typing import List, Optional
from detector import Detection


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


class TargetTracker:
    """
    타겟 고정 + IoU 추적.

    사용:
        tracker = TargetTracker(miss_timeout=1.5, iou_thresh=0.3, max_dist=200)
        target = tracker.update(detections)
    """

    def __init__(self,
                 miss_timeout: float = 1.5,
                 iou_thresh: float = 0.3,
                 max_dist: float = 200):
        self._target      = None   # 현재 고정된 타겟
        self._last_seen   = 0.0    # 마지막으로 타겟 탐지된 시각
        self._miss_timeout = miss_timeout
        self._iou_thresh   = iou_thresh
        self._max_dist     = max_dist

    @property
    def target(self) -> Optional[Detection]:
        return self._target

    @property
    def miss_elapsed(self) -> float:
        if self._target is None:
            return 0.0
        return time.time() - self._last_seen

    def update(self, detections: List[Detection]) -> Optional[Detection]:
        """
        매 프레임 호출.
        현재 타겟을 추적하거나 새 타겟을 선택해서 반환.
        """
        monsters = [d for d in detections if d.class_id == 0]
        now = time.time()

        # ── 타겟 없음 → 새 타겟 선택 ─────────────────────────
        if self._target is None:
            new = select_by_confidence(monsters)
            if new:
                self._target    = new
                self._last_seen = now
                print(f"[Tracker] 타겟 선택: cx={new.cx} cy={new.cy} "
                      f"conf={new.confidence:.2f}")
            return self._target

        # ── 타겟 있음 → IoU + 거리로 추적 ────────────────────
        best      = None
        best_score = -1.0

        for d in monsters:
            iou  = calc_iou(self._target, d)
            dist = math.hypot(d.cx - self._target.cx,
                              d.cy - self._target.cy)

            # IoU 또는 거리 기준 매칭
            if iou >= self._iou_thresh or dist <= self._max_dist:
                score = iou + (1.0 - min(dist, self._max_dist) / self._max_dist)
                if score > best_score:
                    best_score = score
                    best = d

        if best is not None:
            # 타겟 갱신 (이동 추적)
            self._target    = best
            self._last_seen = now
        else:
            # 탐지 안 됨 → miss_timeout 체크
            elapsed = now - self._last_seen
            if elapsed > self._miss_timeout:
                print(f"[Tracker] 타겟 소실 ({elapsed:.1f}s) → 새 타겟 탐색")
                self._target = None
                # 이번 프레임에 새 타겟 바로 선택
                new = select_by_confidence(monsters)
                if new:
                    self._target    = new
                    self._last_seen = now
                    print(f"[Tracker] 새 타겟: cx={new.cx} cy={new.cy} "
                          f"conf={new.confidence:.2f}")

        return self._target

    def reset(self):
        """타겟 초기화."""
        self._target    = None
        self._last_seen = 0.0
