"""
level_detector.py
-----------------
레벨 / HP 인식 모듈.

인식 구조 (레퍼런스 hp_reader.py / level_reader.py 기반):

[HP 인식]
  - HP 바 색상: 파란색 (실측 RGB≈(0,36~43,175~212) → HSV Hue≈220~230°)
    OpenCV HSV에서 Hue는 0~180 스케일이므로 220~230° → H=110~130
  - HSV 범위: H=100~140 (파란색 계열, S/V 하한으로 배경 제외)
  - 핵심: "HP : 115 / 115" 텍스트가 바 중간을 가로막아 연속 열이 끊김
    → 연속 열(break at first gap) 방식 사용 불가
    → 전체 파란 열 합계 비율 방식 사용 (np.count_nonzero)
  - 결과: 0.0 ~ 100.0 (%)

[레벨 인식]
  - 엔진: easyocr (detail=1, confidence 포함)
  - 전처리: 2배 확대 + CLAHE(clipLimit=3.0) + OTSU 이진화
  - 신뢰도 임계값: 0.1 (LEV:14 등 OCR 신뢰도 낮아 완화)
  - 정규식 패턴:
      1) LEV/LEC/LEU 계열: [Ll][Ee][VvCcUu][: .]*?(\\d+)  ← V→C/U 오인식 포함
      2) Lv.숫자 계열:     [Ll][Vv]\\.?\\s*(\\d+)
      3) 숫자만:           ^\\d+$ (1~99 범위)

사용:
    ld = LevelDetector(level_roi, hp_roi)
    level = ld.read_level(frame_bgr)   # int or None
    hp    = ld.read_hp(frame_bgr)      # 0.0~100.0 (%) or None
"""

import re
import time
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
#  HP 바 HSV 범위
#  실측: RGB≈(0, 36~43, 175~212) → HSV Hue≈220~230°
#  OpenCV HSV: Hue는 0~180 스케일 (실제각도 /2)
#  → 220~230° / 2 = 110~115 → 여유분 포함 H=100~140
# ─────────────────────────────────────────────────────────────
_HP_HSV_RANGES = [
    # 파란색 (H=100~140, S=60+, V=30+ — 어두운 파란도 포함)
    ((100, 60, 30), (140, 255, 255)),
]


def _calc_hp_pct(crop_bgr: np.ndarray) -> float:
    """HP 바 크롭 이미지에서 HP%를 계산합니다.

    실측 기반:
      - 각 열에 파란 픽셀이 하나라도 있으면 "채워진 열"로 판단
      - HP 바 중간에 텍스트("HP : 115 / 115")가 있어 연속 열이 끊김
        → 왼쪽부터 연속 방식(break)은 사용 불가
        → 전체 파란 열 합계(count_nonzero) 비율로 계산

    Args:
        crop_bgr: BGR 이미지 (HP 바 영역)

    Returns:
        HP% (0.0 ~ 100.0)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 100.0

    # BGR → HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # 파란색 픽셀 마스크 (HSV H=100~140)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in _HP_HSV_RANGES:
        m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                             np.array(upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, m)

    total_cols = mask.shape[1]
    if total_cols == 0:
        return 100.0

    # 각 열에 파란 픽셀이 하나라도 있으면 True (axis=0 → 열 방향 축소)
    col_has_blue = np.any(mask > 0, axis=0)   # shape: (width,)

    # ★ 핵심: 연속이 아닌 전체 파란 열 합계 비율
    blue_cols = int(np.count_nonzero(col_has_blue))
    hp_pct = round((blue_cols / total_cols) * 100.0, 1)

    return hp_pct


def _preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """레벨 OCR 인식률을 높이기 위한 전처리.

    실측 기반:
      - 배경: 주황/갈색 그라데이션 (어두운 계열)
      - 텍스트: 흰색 (RGB≈240,245,255) → 그레이스케일 시 배경보다 밝음
      → OTSU 이진화(THRESH_BINARY)로 흰 텍스트=255, 어두운 배경=0
      → easyocr은 밝은 배경+어두운 텍스트를 선호하므로 추가 반전

    처리 순서:
      1) 3배 확대 (작은 텍스트 인식률 향상)
      2) 그레이스케일
      3) CLAHE 대비 향상
      4) OTSU 이진화 (흰 텍스트 추출)
      5) bitwise_not 반전 → 흰 배경 + 검정 텍스트 (easyocr 최적)
    """
    h, w = crop_bgr.shape[:2]
    # 3배 확대 (30px 높이 → 90px, OCR 인식률 향상)
    enlarged = cv2.resize(crop_bgr, (w * 3, h * 3),
                          interpolation=cv2.INTER_LINEAR)

    # 그레이스케일
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # CLAHE 대비 향상
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # OTSU 이진화: 흰 텍스트(밝음)=255, 주황/갈색 배경(어두움)=0
    _, binary = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # easyocr 최적: 흰 배경 + 검정 텍스트로 반전
    return cv2.bitwise_not(binary)


def _parse_level(text: str):
    """OCR 텍스트에서 레벨 숫자를 추출합니다.

    레퍼런스 구조 (level_reader.py _parse_level):
      1) LEV/LEC/LEU 계열 (V→C/U 오인식 포함, 대소문자 무관)
         예: "LEV:14", "LEC:14", "LEU:14", "Lev:5"
      2) Lv.숫자 계열
         예: "Lv.5", "Lv 5", "LV.5", "lv5"
      3) 숫자만 (region이 레벨 숫자만 보이는 경우, 1~99)
    """
    text_stripped = text.strip()

    # 1) LEV / LEC / LEU 계열
    m = re.search(r"[Ll][Ee][VvCcUu][: .]*?(\d+)", text_stripped)
    if m:
        return int(m.group(1))

    # 2) Lv.숫자 계열
    m = re.search(r"[Ll][Vv]\.?\s*(\d+)", text_stripped)
    if m:
        return int(m.group(1))

    # 3) 숫자만
    m = re.search(r"^\d+$", text_stripped.replace(" ", ""))
    if m:
        val = int(m.group(0))
        if 1 <= val <= 99:
            return val

    return None


# easyocr 글로벌 싱글톤 (최초 1회만 초기화)
_ocr_reader = None


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("[LevelDetector] easyocr 초기화 중...")
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[LevelDetector] easyocr 초기화 완료")
        except ImportError:
            print("[LevelDetector] easyocr 미설치 → pip install easyocr")
            raise
    return _ocr_reader


class LevelDetector:
    """레벨 OCR + HP 바 인식 클래스.

    레퍼런스 HpReader + LevelReader 구조를 단일 클래스로 통합.
    """

    def __init__(self,
                 level_roi: dict = None,
                 hp_roi: dict = None,
                 hp_color: str = "blue",     # 하위 호환용 (무시됨, 항상 파란색)
                 target_level: int = 5):
        self._level_roi    = level_roi
        self._hp_roi       = hp_roi
        self._target_level = target_level

        # easyocr lazy init
        self._ocr = None

        # 캐시
        self._last_level:    int   = None
        self._last_hp:       float = None
        self._last_level_t:  float = 0.0
        self._last_hp_t:     float = 0.0
        self._level_interval = 2.0   # 레벨: 2초마다 OCR
        self._hp_interval    = 0.5   # HP: 0.5초마다

        print("[LevelDetector] 초기화 완료 (HP: 파란색 HSV H=100~140, 전체 열 합계 방식)")

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = _get_ocr()

    def update_rois(self, level_roi=None, hp_roi=None):
        if level_roi is not None:
            self._level_roi = level_roi
        if hp_roi is not None:
            self._hp_roi = hp_roi

    # ─────────────────────────────────────────────────────────
    #  ROI 크롭 (공통)
    # ─────────────────────────────────────────────────────────

    def _crop_roi(self, frame_bgr: np.ndarray, roi: dict):
        """ROI dict {"x","y","w","h"} 또는 {"x","y","width","height"} 지원."""
        if frame_bgr is None or roi is None:
            return None
        x = roi.get("x", 0)
        y = roi.get("y", 0)
        w = roi.get("w", roi.get("width", 0))
        h = roi.get("h", roi.get("height", 0))
        if w <= 0 or h <= 0:
            return None
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
        """레벨 ROI OCR → 레벨 숫자 반환.

        레퍼런스 LevelReader.read() 구조:
          - detail=1 (신뢰도 포함)
          - 신뢰도 < 0.1 은 무시 (완화된 임계값)
          - _parse_level로 LEV/LEC/LEU/Lv 패턴 파싱

        캐시: 2초
        """
        now = time.monotonic()
        if now - self._last_level_t < self._level_interval:
            return self._last_level
        self._last_level_t = now

        if self._level_roi is None:
            return self._last_level

        crop = self._crop_roi(frame_bgr, self._level_roi)
        if crop is None:
            return self._last_level

        # 전처리 (2배 확대 + CLAHE + OTSU)
        processed = _preprocess_for_ocr(crop)

        # OCR (detail=1 → (bbox, text, confidence) 형태)
        try:
            self._ensure_ocr()
            results = self._ocr.readtext(processed, detail=1, paragraph=False)
        except Exception as e:
            print(f"[LevelDetector] easyocr 오류: {e}")
            return self._last_level

        # 결과 파싱 (레퍼런스: confidence < 0.1 무시)
        for (_, text, confidence) in results:
            if confidence < 0.1:   # ← 레퍼런스와 동일, 완화된 임계값
                continue
            level = _parse_level(text)
            if level is not None and 1 <= level <= 99:
                if level != self._last_level:
                    print(f"[LevelDetector] 레벨: {self._last_level} → {level}"
                          f"  (OCR='{text}', conf={confidence:.2f})")
                self._last_level = level
                return level

        # 인식 실패 → 캐시 유지
        debug_list = [(t, f"{c:.2f}") for (_, t, c) in results]
        print(f"[LevelDetector] 레벨 인식 실패. OCR={debug_list}")
        return self._last_level

    def is_target_level_reached(self, frame_bgr: np.ndarray) -> bool:
        lv = self.read_level(frame_bgr)
        if lv is None:
            return False
        return lv >= self._target_level

    # ─────────────────────────────────────────────────────────
    #  HP 인식
    # ─────────────────────────────────────────────────────────

    def read_hp(self, frame_bgr: np.ndarray) -> float:
        """HP ROI에서 파란색 픽셀 열 비율로 HP% 계산.

        레퍼런스 구조 (hp_reader.py):
          - HSV 파란색 범위 (H=100~140, 실측 RGB≈(0,36~43,175~212))
          - 텍스트가 바 중간을 가로막으므로 연속 열 방식 불가
          - 전체 파란 열 합계(count_nonzero) 비율 사용

        캐시: 0.5초

        Returns:
            0.0 ~ 100.0 (%) or None (ROI 없음)
        """
        now = time.monotonic()
        if now - self._last_hp_t < self._hp_interval:
            return self._last_hp
        self._last_hp_t = now

        if self._hp_roi is None:
            return self._last_hp

        crop = self._crop_roi(frame_bgr, self._hp_roi)
        if crop is None:
            return self._last_hp

        hp_pct = _calc_hp_pct(crop)

        # 5% 이상 변화 시 로그
        if self._last_hp is None or abs(hp_pct - self._last_hp) >= 5.0:
            print(f"[LevelDetector] HP: {self._last_hp}% → {hp_pct:.1f}%")

        self._last_hp = hp_pct
        return self._last_hp

    def is_hp_low(self, frame_bgr: np.ndarray, threshold: float = 50.0) -> bool:
        """HP가 threshold% 미만이면 True.

        Args:
            threshold: 기준값 (기본 50.0%)
                       ※ 이전 버전은 0~1 비율이었으나, 0~100%로 변경됨
        """
        hp = self.read_hp(frame_bgr)
        if hp is None:
            return False
        return hp < threshold
