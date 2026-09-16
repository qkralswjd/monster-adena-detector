"""
level_detector.py
-----------------
레벨 UI / HP바 영역을 픽셀 분석으로 인식.

동작 원리:
  - 레벨 ROI: OCR(pytesseract) 또는 숫자 이미지 템플릿 매칭으로 레벨 숫자 읽기
    → pytesseract 없으면 easyocr fallback
  - HP ROI: 빨간(또는 설정된) 픽셀 비율로 HP% 계산
    → 예) 전체 픽셀 중 HP 색상 픽셀 비율 = HP%

사용:
    ld = LevelDetector(level_roi, hp_roi, hp_color="red")
    level = ld.read_level(frame_bgr)   # int or None
    hp    = ld.read_hp(frame_bgr)      # 0.0~1.0 or None

frame_bgr: mss 캡처 → numpy array (전체 모니터 캡처 기준)
"""

import numpy as np
import cv2
import time


# HP바 색상 범위 (HSV)
# 빨간색 두 범위 (Hue 0~10, 170~180)
HP_HSV_RANGES = [
    ((0,   80, 80),  (10,  255, 255)),
    ((170, 80, 80),  (180, 255, 255)),
]


class LevelDetector:

    def __init__(self,
                 level_roi: dict = None,
                 hp_roi: dict = None,
                 hp_color: str = "red",
                 target_level: int = 5):
        """
        level_roi  : {"x","y","w","h"} 절대 화면 좌표
        hp_roi     : {"x","y","w","h"} 절대 화면 좌표
        hp_color   : "red" (기본) — 나중에 다른 색 지원 가능
        target_level: 이 레벨 달성 시 DUMMY_ATTACK 종료
        """
        self._level_roi    = level_roi
        self._hp_roi       = hp_roi
        self._hp_color     = hp_color
        self._target_level = target_level

        # OCR 엔진 초기화
        self._ocr_engine = None
        self._init_ocr()

        # 캐시 (너무 자주 호출 방지)
        self._last_level     = None
        self._last_hp        = None
        self._last_level_t   = 0.0
        self._last_hp_t      = 0.0
        self._level_interval = 2.0   # 레벨은 2초마다 읽기
        self._hp_interval    = 0.3   # HP는 0.3초마다 읽기

    def _init_ocr(self):
        """OCR 엔진 초기화 (pytesseract → easyocr 순서)."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._ocr_engine = "tesseract"
            print("[LevelDetector] OCR: pytesseract")
        except Exception:
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._ocr_engine = "easyocr"
                print("[LevelDetector] OCR: easyocr")
            except Exception:
                self._ocr_engine = None
                print("[LevelDetector] OCR 엔진 없음 → 레벨 인식 불가 (숫자만 인식)")

    def update_rois(self, level_roi=None, hp_roi=None):
        """config reload 시 ROI 갱신."""
        if level_roi:
            self._level_roi = level_roi
        if hp_roi:
            self._hp_roi = hp_roi

    # ─────────────────────────────────────────────────────────
    #  전체 화면 캡처 → ROI 크롭
    # ─────────────────────────────────────────────────────────

    def _crop_roi(self, frame_bgr: np.ndarray, roi: dict):
        """
        frame_bgr: 전체 모니터 캡처 (mss BGRA → BGR 변환된 것)
        roi: {"x","y","w","h"} 절대 화면 좌표
        → ROI 영역 크롭 반환
        """
        if frame_bgr is None or roi is None:
            return None
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    # ─────────────────────────────────────────────────────────
    #  레벨 인식
    # ─────────────────────────────────────────────────────────

    def read_level(self, frame_bgr: np.ndarray) -> int:
        """
        레벨 ROI에서 숫자 OCR.
        캐시: 2초마다 갱신.
        반환: int or None
        """
        now = time.monotonic()
        if now - self._last_level_t < self._level_interval:
            return self._last_level
        self._last_level_t = now

        if self._level_roi is None:
            return None

        crop = self._crop_roi(frame_bgr, self._level_roi)
        if crop is None:
            return None

        level = self._ocr_digit(crop)
        if level is not None and 1 <= level <= 99:
            self._last_level = level
        return self._last_level

    def _ocr_digit(self, crop_bgr: np.ndarray) -> int:
        """이미지에서 숫자 하나 읽기."""
        # 전처리: 그레이 → 이진화
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # 크기 확대 (OCR 정확도 향상)
        scale = 3
        h, w = gray.shape
        gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = None

        if self._ocr_engine == "tesseract":
            try:
                import pytesseract
                cfg = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(binary, config=cfg).strip()
            except Exception as e:
                print(f"[LevelDetector] tesseract 오류: {e}")

        elif self._ocr_engine == "easyocr":
            try:
                results = self._reader.readtext(binary, detail=0,
                                                allowlist="0123456789")
                text = "".join(results).strip()
            except Exception as e:
                print(f"[LevelDetector] easyocr 오류: {e}")

        if text:
            digits = "".join(c for c in text if c.isdigit())
            if digits:
                try:
                    return int(digits[:2])  # 최대 2자리
                except ValueError:
                    pass
        return None

    def is_target_level_reached(self, frame_bgr: np.ndarray) -> bool:
        """목표 레벨 달성 여부."""
        lv = self.read_level(frame_bgr)
        if lv is None:
            return False
        return lv >= self._target_level

    # ─────────────────────────────────────────────────────────
    #  HP 인식
    # ─────────────────────────────────────────────────────────

    def read_hp(self, frame_bgr: np.ndarray) -> float:
        """
        HP ROI에서 HP 비율 계산.
        캐시: 0.3초마다 갱신.
        반환: 0.0~1.0 or None
        """
        now = time.monotonic()
        if now - self._last_hp_t < self._hp_interval:
            return self._last_hp
        self._last_hp_t = now

        if self._hp_roi is None:
            return None

        crop = self._crop_roi(frame_bgr, self._hp_roi)
        if crop is None:
            return None

        hp = self._calc_hp_ratio(crop)
        self._last_hp = hp
        return hp

    def _calc_hp_ratio(self, crop_bgr: np.ndarray) -> float:
        """
        HP 색상 픽셀 비율 계산.
        빨간색 픽셀 수 / 전체 픽셀 수 = HP%
        (HP바가 왼쪽→오른쪽으로 차있는 구조 가정)
        """
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for (lo, hi) in HP_HSV_RANGES:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask |= cv2.inRange(hsv, lo_arr, hi_arr)

        total = mask.size
        filled = int(np.count_nonzero(mask))

        if total == 0:
            return None

        ratio = filled / total
        # 클램프
        ratio = max(0.0, min(1.0, ratio))
        return ratio

    def is_hp_low(self, frame_bgr: np.ndarray, threshold: float = 0.5) -> bool:
        """HP가 threshold 미만인지 확인."""
        hp = self.read_hp(frame_bgr)
        if hp is None:
            return False
        return hp < threshold
