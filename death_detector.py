"""
death_detector.py
-----------------
타겟 몬스터의 사망/소실 감지 모듈.

판정 기준:
  1. 현재 프레임 탐지 목록에서 이전 타겟 위치와 IoU >= thresh 인 박스가 없음
  2. 위 상태가 death_timeout_sec 이상 지속되면 → 사망 확정

사용법:
    dd = DeathDetector(timeout_sec=1.5, iou_thresh=0.3)
    dd.reset()                          # 새 타겟 지정 시 초기화
    is_dead = dd.update(detections, current_target)
"""

import time
from typing import List, Optional
from detector import Detection
from target_selector import find_matching


class DeathDetector:
    def __init__(self, timeout_sec: float = 1.5, iou_thresh: float = 0.3):
        self._timeout   = timeout_sec
        self._iou_thresh = iou_thresh
        self._miss_since: Optional[float] = None  # 소실 시작 시각

    def reset(self):
        """새 타겟 지정 시 반드시 호출."""
        self._miss_since = None

    def update(self, detections: List[Detection],
               target: Optional[Detection]) -> bool:
        """
        Returns:
            True  → 타겟 사망/소실 확정 (새 타겟 선택 필요)
            False → 타겟 살아있음
        """
        if target is None:
            return True  # 타겟 자체가 없으면 사망으로 취급

        matched = find_matching(detections, target, self._iou_thresh)

        if matched is not None:
            # 탐지됨 → 살아있음
            self._miss_since = None
            return False
        else:
            # 탐지 안됨 → 소실 타이머 시작
            now = time.time()
            if self._miss_since is None:
                self._miss_since = now

            elapsed = now - self._miss_since
            if elapsed >= self._timeout:
                self._miss_since = None
                return True  # 사망 확정
            return False

    @property
    def miss_elapsed(self) -> float:
        """현재 소실 중이면 경과 시간(초), 아니면 0."""
        if self._miss_since is None:
            return 0.0
        return time.time() - self._miss_since
