"""아데나(전리품) 텍스트 탐지 모듈.

몬스터가 죽은 후 바닥에 나타나는 "아데나" 텍스트를
HSV 색상 필터로 탐지하여 클릭 좌표를 반환합니다.

탐지 전략:
    1. HSV 색상 필터로 아데나 이름표 후보 영역(흰 테두리) 추출
    2. dilation으로 이름표 전체 박스 연결
    3. 크기 필터로 오탐 제거
    → 전체 화면 OCR 없이 HSV만으로 즉시 반환 (속도/정확도 향상)

사용법:
    detector = LootDetector(
        scan_region={"x": 360, "y": 240, "width": 1200, "height": 500},
        loot_keywords=["아데나", "Adena"],
    )
    loots = detector.find(frame)
    # loots = [(center_x, center_y, "adena", 1.0), ...]
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("loot_detector")

# ── 리니지 클래식 아데나 이름표 색상 범위 ──────────────────────────────────
# 흰 테두리: HSV S<50, V>200
_ADENA_WHITE_LOWER = (0,   0,   200)
_ADENA_WHITE_UPPER = (180, 50,  255)
_ADENA_DILATION_K  = 7    # dilation 커널 크기
_ADENA_DILATION_IT = 3    # dilation 반복 횟수
_ADENA_BOX_MIN_W   = 40   # 최소 박스 너비 (px)
_ADENA_BOX_MIN_H   = 15   # 최소 박스 높이 (px)
_ADENA_BOX_MAX_W   = 300  # 최대 박스 너비
_ADENA_BOX_MAX_H   = 80   # 최대 박스 높이


def _extract_candidate_boxes(
    crop_bgr: np.ndarray,
    padding: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """흰 테두리 HSV 마스크 + dilation으로 아데나 이름표 후보 박스 추출."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _ADENA_WHITE_LOWER, _ADENA_WHITE_UPPER)

    # dilation: 얇은 테두리 선들을 연결
    kernel  = np.ones((_ADENA_DILATION_K, _ADENA_DILATION_K), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=_ADENA_DILATION_IT)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_crop, w_crop = crop_bgr.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < _ADENA_BOX_MIN_W or h < _ADENA_BOX_MIN_H:
            continue
        if w > _ADENA_BOX_MAX_W or h > _ADENA_BOX_MAX_H:
            continue
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_crop, x + w + padding)
        y2 = min(h_crop, y + h + padding)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


class LootDetector:
    """화면에서 아이템 이름(아데나 등) 텍스트를 탐지합니다."""

    def __init__(
        self,
        scan_region: dict,
        loot_keywords: list = None,
        scan_interval_s: float = 0.3,
        min_confidence: float  = 0.4,
    ):
        """
        Args:
            scan_region: 스캔할 ROI 영역 {"x","y","width","height"}
            loot_keywords: 탐지할 텍스트 목록 (기본: ["아데나","Adena","adena"])
            scan_interval_s: 스캔 최소 간격 (초)
            min_confidence: (HSV 방식에서는 미사용, 하위 호환용)
        """
        self.scan_region    = scan_region
        self.loot_keywords  = loot_keywords or ["아데나", "데나", "Adena", "adena", "ADENA"]
        self.scan_interval  = scan_interval_s
        self.min_confidence = min_confidence

        self._last_scan_time = 0.0
        self._cached_loots: list = []

    def invalidate(self) -> None:
        """캐시를 무효화합니다. 다음 find() 호출 시 즉시 재스캔합니다."""
        self._last_scan_time = 0.0
        self._cached_loots   = []
        logger.debug("[LootDetector] 캐시 무효화")

    def find(self, frame: np.ndarray) -> List[Tuple[int, int, str, float]]:
        """프레임에서 아데나 이름표를 HSV 박스 탐지로 찾습니다.

        Returns:
            [(screen_x, screen_y, "adena", 1.0), ...]
        """
        now = time.time()
        if now - self._last_scan_time < self.scan_interval:
            return self._cached_loots

        self._last_scan_time = now

        # 스캔 영역 크롭
        rx = self.scan_region.get("x", 0)
        ry = self.scan_region.get("y", 0)
        rw = self.scan_region.get("width",  frame.shape[1])
        rh = self.scan_region.get("height", frame.shape[0])
        rw = min(rw, frame.shape[1] - rx)
        rh = min(rh, frame.shape[0] - ry)
        crop = frame[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            return self._cached_loots

        # HSV 흰 테두리 박스 탐지
        boxes = _extract_candidate_boxes(crop)

        if not boxes:
            logger.debug("[LootDetector] 후보 박스 없음")
            self._cached_loots = []
            return self._cached_loots

        # 박스 중심 → 화면 절대좌표 변환
        loots = []
        for (bx, by, bw, bh) in boxes:
            screen_x = bx + bw // 2 + rx
            screen_y = by + bh // 2 + ry
            loots.append((screen_x, screen_y, "adena", 1.0))
            logger.info(
                f"[LootDetector] ✅ HSV탐지: at ({screen_x},{screen_y}) "
                f"box={bw}×{bh}"
            )

        self._cached_loots = loots
        return loots

    def find_nearest(
        self,
        frame: np.ndarray,
        ref_x: int = 0,
        ref_y: int = 0,
    ) -> Optional[Tuple[int, int, str, float]]:
        """가장 가까운 아이템을 반환합니다."""
        loots = self.find(frame)
        if not loots:
            return None

        import math
        return min(loots, key=lambda l: math.hypot(l[0] - ref_x, l[1] - ref_y))
