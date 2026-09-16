"""
target_selector.py
------------------
타겟 선택 + 추적 로직.

동작 원칙:
  1. 타겟 없음 → confidence 최고 몬스터 선택 (고정)
  2. 타겟 고정 후 → 속도 기반으로 같은 몬스터 추적
     - 경과 시간 × max_speed(px/s) = 허용 이동 거리
     - 여러 몬스터 중 가장 가까운 것 선택
  3. miss_timeout 초 이상 탐지 안 되면 → 사망 추정 → 새 타겟 선택

좌표 원칙:
  - 입력/출력 모두 화면 절대좌표 그대로. 변환 없음.
"""

import math
import time
from typing import List, Optional, Tuple
from detector import Detection


def select_by_confidence(detections: List[Detection]) -> Optional[Detection]:
    """confidence 가장 높은 몬스터 반환."""
    monsters = [d for d in detections if d.class_id == 0]
    if not monsters:
        return None
    return max(monsters, key=lambda d: d.confidence)


class TargetTracker:
    """
    타겟 1개 고정 + 속도 기반 추적.

    사용:
        tracker = TargetTracker(miss_timeout=3.5, max_speed=400)
        target, miss_elapsed = tracker.update(detections)

        miss_elapsed == 0.0  → 이번 프레임에 탐지됨
        miss_elapsed  > 0.0  → 탐지 못한 경과 시간 (초)
    """

    def __init__(self,
                 miss_timeout: float = 3.5,
                 max_dist: float = 200,       # 하위 호환용 (미사용)
                 max_speed: float = 250):     # px/초 - 몬스터 최대 이동 속도
        self._target       = None
        self._last_seen    = 0.0
        self._last_seen_t  = 0.0   # 마지막 탐지 시각
        self._miss_timeout = miss_timeout
        self._max_speed    = max_speed  # px/초

    @property
    def target(self) -> Optional[Detection]:
        return self._target

    def update(self, detections: List[Detection]) -> Tuple[Optional[Detection], float]:
        """
        매 프레임 호출.
        반환: (현재 타겟, miss_elapsed)
          - miss_elapsed == 0.0 : 이번 프레임에 탐지됨
          - miss_elapsed  > 0.0 : 마지막 탐지 후 경과 초
        """
        monsters = [d for d in detections if d.class_id == 0]
        now = time.time()

        # ── 타겟 없음 → 새 타겟 선택 ─────────────────────────
        if self._target is None:
            new = select_by_confidence(monsters)
            if new:
                self._target     = new
                self._last_seen  = now
                self._last_seen_t = now
                print(f"[Tracker] ★ 타겟 선택: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self._target, 0.0

        # ── 타겟 있음 → 속도 기반으로 같은 몬스터 탐색 ──────
        # 마지막 탐지 이후 경과 시간만큼 이동 허용
        elapsed_since_seen = now - self._last_seen
        max_allowed_dist   = self._max_speed * max(elapsed_since_seen, 0.05)

        best      = None
        best_dist = max_allowed_dist

        for d in monsters:
            dist = math.hypot(d.cx - self._target.cx,
                              d.cy - self._target.cy)
            if dist < best_dist:
                best_dist = dist
                best = d

        if best is not None:
            # 탐지 성공 → 위치 갱신
            prev_cx, prev_cy = self._target.cx, self._target.cy
            self._target     = best
            self._last_seen  = now
            self._last_seen_t = now
            moved = math.hypot(best.cx - prev_cx, best.cy - prev_cy)
            if moved > 30:
                print(f"[Tracker] 이동: ({prev_cx},{prev_cy}) → ({best.cx},{best.cy})  Δ{moved:.0f}px")
            return self._target, 0.0

        # ── 탐지 실패 → miss 타이머 체크 ─────────────────────
        elapsed = now - self._last_seen

        if elapsed > self._miss_timeout:
            # 사망 추정 → 타겟 해제
            print(f"[Tracker] ✗ 소실 ({elapsed:.1f}s) → 사망 추정, 새 타겟 탐색")
            self._target = None

            # 이번 프레임 새 타겟 즉시 선택
            new = select_by_confidence(monsters)
            if new:
                self._target     = new
                self._last_seen  = now
                self._last_seen_t = now
                print(f"[Tracker] ★ 새 타겟: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self._target, 0.0

        # miss 중이지만 timeout 전 → 마지막 타겟 유지
        return self._target, elapsed

    def reset(self):
        """타겟 초기화."""
        self._target    = None
        self._last_seen = 0.0
        print("[Tracker] 리셋")
