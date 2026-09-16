"""
level_detector.py
-----------------
레벨 / HP 텍스트 OCR 인식.

동작 원리:
  - 레벨 ROI: "LEV:15" 형태 텍스트 → 숫자 추출
  - HP ROI:   "HP:115/115" 형태 텍스트 → 현재HP/최대HP 비율 계산

OCR 엔진 우선순위:
  1. pytesseract (설치된 경우)
  2. easyocr (fallback)
  3. 없으면 None 반환

사용:
    ld = LevelDetector(level_roi, hp_roi)
    level = ld.read_level(frame_bgr)   # int or None
    hp    = ld.read_hp(frame_bgr)      # 0.0~1.0 or None
"""

import re
import time
import numpy as np
import cv2


class LevelDetector:

    def __init__(self,
                 level_roi: dict = None,
                 hp_roi: dict = None,
                 hp_color: str = "red",      # 호환성 유지용 (사용 안 함)
                 target_level: int = 5):
        self._level_roi    = level_roi
        self._hp_roi       = hp_roi
        self._target_level = target_level

        # OCR 엔진 초기화
        self._ocr_engine = None
        self._reader     = None
        self._init_ocr()

        # 캐시
        self._last_level   = None
        self._last_hp      = None
        self._last_level_t = 0.0
        self._last_hp_t    = 0.0
        self._level_interval = 2.0   # 레벨은 2초마다
        self._hp_interval    = 0.5   # HP는 0.5초마다

    # ─────────────────────────────────────────────────────────
    #  OCR 초기화
    # ─────────────────────────────────────────────────────────

    def _init_ocr(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._ocr_engine = "tesseract"
            print("[LevelDetector] OCR: pytesseract")
            return
        except Exception:
            pass

        try:
            import easyocr
            self._reader     = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._ocr_engine = "easyocr"
            print("[LevelDetector] OCR: easyocr")
            return
        except Exception:
            pass

        self._ocr_engine = None
        print("[LevelDetector] OCR 엔진 없음 → pip install easyocr 권장")

    def update_rois(self, level_roi=None, hp_roi=None):
        if level_roi:
            self._level_roi = level_roi
        if hp_roi:
            self._hp_roi = hp_roi

    # ─────────────────────────────────────────────────────────
    #  ROI 크롭
    # ─────────────────────────────────────────────────────────

    def _crop_roi(self, frame_bgr: np.ndarray, roi: dict):
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
    #  이미지 전처리 (OCR 정확도 향상)
    # ─────────────────────────────────────────────────────────

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        확대 + 이진화.
        리니지 UI 텍스트: 밝은 주황/흰색 글자, 어두운 배경
        """
        # 4배 확대
        h, w = crop_bgr.shape[:2]
        big = cv2.resize(crop_bgr, (w * 4, h * 4),
                         interpolation=cv2.INTER_LINEAR)
        # 그레이 변환
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        # CLAHE (대비 향상)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # ─────────────────────────────────────────────────────────
    #  OCR 텍스트 추출
    # ─────────────────────────────────────────────────────────

    def _ocr_text(self, crop_bgr: np.ndarray) -> str:
        """이미지에서 텍스트 추출 (숫자 + 콜론 + 슬래시 허용)."""
        binary = self._preprocess(crop_bgr)

        if self._ocr_engine == "tesseract":
            try:
                import pytesseract
                # 숫자, 콜론, 슬래시, 알파벳 허용
                cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:/LEVHPlevhp"
                return pytesseract.image_to_string(binary, config=cfg).strip()
            except Exception as e:
                print(f"[LevelDetector] tesseract 오류: {e}")

        elif self._ocr_engine == "easyocr":
            try:
                results = self._reader.readtext(
                    binary, detail=0,
                    allowlist="0123456789:/LEVHPlevhp"
                )
                return " ".join(results).strip()
            except Exception as e:
                print(f"[LevelDetector] easyocr 오류: {e}")

        return ""

    # ─────────────────────────────────────────────────────────
    #  레벨 인식
    # ─────────────────────────────────────────────────────────

    def read_level(self, frame_bgr: np.ndarray) -> int:
        """
        레벨 ROI OCR → "LEV:15" or "15" 에서 숫자 추출.
        캐시: 2초
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

        text = self._ocr_text(crop)
        level = self._parse_level(text)

        if level is not None and 1 <= level <= 99:
            self._last_level = level
            print(f"[LevelDetector] 레벨 인식: {level}  (OCR: '{text}')")

        return self._last_level

    def _parse_level(self, text: str) -> int:
        """
        "LEV:15", "LEV 15", "15" 등에서 레벨 숫자 추출.
        """
        if not text:
            return None
        # 숫자만 추출
        digits = re.findall(r'\d+', text)
        if not digits:
            return None
        # 가장 처음 나오는 숫자 (1~99 범위)
        for d in digits:
            v = int(d)
            if 1 <= v <= 99:
                return v
        return None

    def is_target_level_reached(self, frame_bgr: np.ndarray) -> bool:
        lv = self.read_level(frame_bgr)
        if lv is None:
            return False
        return lv >= self._target_level

    # ─────────────────────────────────────────────────────────
    #  HP 인식
    # ─────────────────────────────────────────────────────────

    def read_hp(self, frame_bgr: np.ndarray) -> float:
        """
        HP ROI OCR → "HP:115/115" 에서 현재HP/최대HP 비율 계산.
        캐시: 0.5초
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

        text = self._ocr_text(crop)
        hp   = self._parse_hp(text)

        if hp is not None:
            self._last_hp = hp
            print(f"[LevelDetector] HP 인식: {hp*100:.0f}%  (OCR: '{text}')")

        return self._last_hp

    def _parse_hp(self, text: str) -> float:
        """
        "HP:115/115", "115/115", "115 / 115" 등에서 현재HP/최대HP 비율 반환.
        """
        if not text:
            return None
        # "숫자/숫자" 패턴
        m = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if m:
            cur = int(m.group(1))
            mx  = int(m.group(2))
            if mx > 0:
                ratio = max(0.0, min(1.0, cur / mx))
                return ratio
        return None

    def is_hp_low(self, frame_bgr: np.ndarray, threshold: float = 0.5) -> bool:
        hp = self.read_hp(frame_bgr)
        if hp is None:
            return False
        return hp < threshold
