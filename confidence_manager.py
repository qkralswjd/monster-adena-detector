"""
confidence_manager.py
---------------------
Target의 추적 신뢰도(confidence)를 관리한다.

- 탐지될 때마다 confidence 증가
- 탐지 안 될 때마다 confidence 감소
- confidence < min_threshold → lost 상태
- lost 상태에서 일정 시간 후 → confirmed lost
"""

import time


class ConfidenceManager:
    def __init__(self,
                 decay: float = 0.05,
                 boost: float = 0.1,
                 min_confidence: float = 0.3,
                 lost_confirm_sec: float = 3.0):
        self._decay = decay
        self._boost = boost
        self._min_conf = min_confidence
        self._lost_confirm_sec = lost_confirm_sec

        self._confidence: float = 1.0
        self._lost_since: float = 0.0
        self._is_lost: bool = False
        self._is_confirmed_lost: bool = False

    def on_detected(self, det_confidence: float = 1.0):
        """탐지 성공 시 호출."""
        self._confidence = min(1.0, self._confidence + self._boost * det_confidence)
        self._is_lost = False
        self._lost_since = 0.0
        self._is_confirmed_lost = False

    def on_missed(self):
        """탐지 실패 시 호출."""
        self._confidence = max(0.0, self._confidence - self._decay)
        if self._confidence < self._min_conf and not self._is_lost:
            self._is_lost = True
            self._lost_since = time.time()
            print(f"[Confidence] Lost 상태 진입 (conf={self._confidence:.2f})")

        if self._is_lost and not self._is_confirmed_lost:
            elapsed = time.time() - self._lost_since
            if elapsed >= self._lost_confirm_sec:
                self._is_confirmed_lost = True
                print(f"[Confidence] Target Lost 확정 ({elapsed:.1f}초)")

    def reset(self, initial: float = 1.0):
        self._confidence = initial
        self._is_lost = False
        self._is_confirmed_lost = False
        self._lost_since = 0.0

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def is_lost(self) -> bool:
        return self._is_lost

    @property
    def is_confirmed_lost(self) -> bool:
        return self._is_confirmed_lost

    @property
    def lost_duration(self) -> float:
        if self._is_lost and self._lost_since > 0:
            return time.time() - self._lost_since
        return 0.0
