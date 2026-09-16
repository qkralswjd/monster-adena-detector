"""
target_selector.py
------------------
타겟 선택 + 추적 로직.

동작 원칙:
  1. 타겟 없음 → confidence 최고 몬스터 선택
  2. 타겟 고정 후 → 속도 벡터 기반 추적
     - 최근 이동 방향/속도를 기억
     - miss 중에는 속도 벡터로 예측 위치 계산 → 박스가 따라감
     - 예측 위치 주변에서 다음 탐지 탐색
  3. miss_timeout 초 이상 탐지 안 되면 → 사망 추정 → 새 타겟 선택

좌표 원칙:
  - 입력/출력 모두 화면 절대좌표 그대로. 변환 없음.
"""

import math
import time
import copy
from typing import List, Optional, Tuple
from detector import Detection


def select_by_confidence(detections: List[Detection]) -> Optional[Detection]:
    """confidence 가장 높은 몬스터 반환."""
    monsters = [d for d in detections if d.class_id == 0]
    if not monsters:
        return None
    return max(monsters, key=lambda d: d.confidence)


def _make_ghost(src: Detection, cx: int, cy: int) -> Detection:
    """src 탐지 결과를 기반으로 cx,cy만 다른 예측 Detection 생성."""
    g = copy.copy(src)
    g.x = cx - src.w // 2
    g.y = cy - src.h // 2
    return g


class TargetTracker:
    """
    타겟 1개 고정 + 속도 벡터 기반 추적.
    miss 중에도 박스가 이동 방향으로 따라감.
    """

    def __init__(self,
                 miss_timeout: float = 3.5,
                 max_dist: float = 150,      # 하위 호환용
                 max_speed: float = 250):    # px/초
        self._target       = None
        self._ghost        = None   # miss 중 예측 위치 Detection
        self._last_seen    = 0.0
        self._miss_timeout = miss_timeout
        self._max_speed    = max_speed

        # 속도 벡터 (px/초)
        self._vx = 0.0
        self._vy = 0.0
        self._prev_t = 0.0

    @property
    def target(self) -> Optional[Detection]:
        """현재 타겟 (miss 중이면 예측 ghost 반환)."""
        return self._ghost if self._ghost is not None else self._target

    def update(self, detections: List[Detection]) -> Tuple[Optional[Detection], float]:
        """
        매 프레임 호출.
        반환: (현재 타겟 or ghost, miss_elapsed)
        """
        monsters = [d for d in detections if d.class_id == 0]
        now = time.time()
        dt  = now - self._prev_t if self._prev_t > 0 else 0.033
        self._prev_t = now

        # ── 타겟 없음 → 새 타겟 선택 ─────────────────────────
        if self._target is None:
            new = select_by_confidence(monsters)
            if new:
                self._target    = new
                self._ghost     = None
                self._last_seen = now
                self._vx = self._vy = 0.0
                print(f"[Tracker] ★ 타겟 선택: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self.target, 0.0

        # ── 현재 예측 위치 계산 (ghost) ───────────────────────
        elapsed = now - self._last_seen
        if elapsed > 0 and (self._vx != 0 or self._vy != 0):
            pred_cx = int(self._target.cx + self._vx * elapsed)
            pred_cy = int(self._target.cy + self._vy * elapsed)
            # 화면 범위 클램핑
            pred_cx = max(0, pred_cx)
            pred_cy = max(0, pred_cy)
            self._ghost = _make_ghost(self._target, pred_cx, pred_cy)
        else:
            self._ghost = None

        # 탐색 기준점: ghost 있으면 ghost, 없으면 마지막 타겟
        ref = self._ghost if self._ghost else self._target
        max_allowed_dist = self._max_speed * min(max(elapsed, 0.05), 1.0)

        # ── 가장 가까운 몬스터 탐색 ───────────────────────────
        best      = None
        best_dist = max_allowed_dist

        for d in monsters:
            dist = math.hypot(d.cx - ref.cx, d.cy - ref.cy)
            if dist < best_dist:
                best_dist = dist
                best = d

        if best is not None:
            # 탐지 성공 → 속도 벡터 갱신 후 위치 업데이트
            prev_cx, prev_cy = self._target.cx, self._target.cy
            if elapsed > 0:
                self._vx = (best.cx - prev_cx) / elapsed
                self._vy = (best.cy - prev_cy) / elapsed
                # 속도 제한
                speed = math.hypot(self._vx, self._vy)
                if speed > self._max_speed:
                    self._vx = self._vx / speed * self._max_speed
                    self._vy = self._vy / speed * self._max_speed

            self._target    = best
            self._ghost     = None
            self._last_seen = now
            moved = math.hypot(best.cx - prev_cx, best.cy - prev_cy)
            if moved > 20:
                print(f"[Tracker] 이동: ({prev_cx},{prev_cy}) → ({best.cx},{best.cy})  Δ{moved:.0f}px")
            return self._target, 0.0

        # ── 탐지 실패 → miss 타이머 체크 ─────────────────────
        if elapsed > self._miss_timeout:
            print(f"[Tracker] ✗ 소실 ({elapsed:.1f}s) → 사망 추정, 새 타겟 탐색")
            self._target = None
            self._ghost  = None
            self._vx = self._vy = 0.0

            new = select_by_confidence(monsters)
            if new:
                self._target    = new
                self._last_seen = now
                print(f"[Tracker] ★ 새 타겟: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self.target, 0.0

        # miss 중 → ghost 위치로 박스 표시
        return self.target, elapsed

    def reset(self):
        self._target = None
        self._ghost  = None
        self._vx = self._vy = 0.0
        self._last_seen = 0.0
        print("[Tracker] 리셋")
