"""
level_detector.py
-----------------
레벨 / HP 인식 모듈.

인식 구조 (레퍼런스 hp_reader.py / level_reader.py 기반):

[HP 인식]
  - HP 바 색상: 빨간색 그라데이션 (실측 확인)
    OpenCV HSV: 빨간색은 H=0~10 + H=170~180 두 구간
  - HSV 범위: H=0~10 (순수 빨강) + H=170~180 (빨강 반대편)
  - 핵심: "HP : 115 / 115" 텍스트가 바 중간을 가로막아 연속 열이 끊김
    → 연속 열(break at first gap) 방식 사용 불가
    → 전체 빨간 열 합계 비율 방식 사용 (np.count_nonzero)
  - 결과: 0.0 ~ 100.0 (%)

[레벨 인식]
  - 엔진: easyocr (detail=1, confidence 포함)
  - 전처리: 2배 확대(INTER_CUBIC) + OTSU 이진화 (실측 conf 0.86, '1' 정확 인식)
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
import threading
import numpy as np
import cv2
import mss as _mss

# ─────────────────────────────────────────────────────────────
#  HP 바 HSV 범위
#  실측: 빨간색 그라데이션 바 (HP:115/115 텍스트 포함)
#  OpenCV HSV: 빨간색은 Hue가 0~10 + 170~180 두 구간
#  → 낮은 채도/명도도 포함 (어두운 빨간 포함)
# ─────────────────────────────────────────────────────────────
_HP_HSV_RANGES = [
    # 빨간색 구간 1: H=0~10 (순수 빨강)
    ((0,  60, 30), (10,  255, 255)),
    # 빨간색 구간 2: H=170~180 (빨강 반대편)
    ((170, 60, 30), (180, 255, 255)),
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

    # 빨간색 픽셀 마스크 (HSV H=0~10 + H=170~180)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in _HP_HSV_RANGES:
        m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                             np.array(upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, m)

    total_cols = mask.shape[1]
    if total_cols == 0:
        return 100.0

    # 각 열에 빨간 픽셀이 하나라도 있으면 True (axis=0 → 열 방향 축소)
    col_has_red = np.any(mask > 0, axis=0)   # shape: (width,)

    # ★ 핵심: 연속이 아닌 전체 빨간 열 합계 비율
    red_cols = int(np.count_nonzero(col_has_red))
    hp_pct = round((red_cols / total_cols) * 100.0, 1)

    return hp_pct


def _preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """레벨 OCR 인식률을 높이기 위한 전처리.

    실측 기반 (debug_preprocess.py 실제 게임 화면 비교 테스트):
      - 배경: 주황/갈색 그라데이션 (어두운 계열)
      - 텍스트: 흰색 (RGB≈240,245,255) → 그레이스케일 시 배경보다 밝음

    비교 결과 (실제 게임 ROI 기준):
      - OTSU만(resize없음): 'Lev:45' ❌ conf=0.38
      - 3x+CLAHE+OTSU:     'Lev:45' ❌ conf=0.23
      - 2x+OTSU:           'Lev:15' ✅ conf=0.86  ← 채택
      - 4x+thresh147:      'Lev;45' ❌ conf=0.40
      - 3x+thresh200:      'LEV:15' ✅ conf=0.47
      - 3x+원본컬러:       'LEv:15' ✅ conf=0.60

    처리 순서:
      1) 2배 확대 (INTER_CUBIC — 부드러운 보간, 1획 보존)
      2) 그레이스케일
      3) OTSU 이진화 (흰 텍스트=255, 어두운 배경=0)
      4) bitwise_not 반전 → 흰 배경 + 검정 텍스트 (easyocr 최적)
    """
    h, w = crop_bgr.shape[:2]

    # 2배 확대 (INTER_CUBIC: 보간 부드럽지만 1획 유지 — 실측 conf 0.86)
    enlarged = cv2.resize(crop_bgr, (w * 2, h * 2),
                          interpolation=cv2.INTER_CUBIC)

    # 그레이스케일
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # OTSU 이진화: 흰 텍스트(밝음)=255, 주황/갈색 배경(어두움)=0
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # easyocr 최적: 흰 배경 + 검정 텍스트로 반전
    return cv2.bitwise_not(binary)


def _parse_level(text: str):
    """OCR 텍스트에서 레벨 숫자를 추출합니다.

    레퍼런스 구조 (level_reader.py _parse_level):
      1) LEV/LEC/LEU 계열 (V→C/U 오인식, ; → : 오인식 포함, 대소문자 무관)
         예: "LEV:14", "LEC:14", "LEU:14", "Lev:5", "Lev ; 15", "Lev ; 15_"
      2) Lv.숫자 계열
         예: "Lv.5", "Lv 5", "LV.5", "lv5"
      3) 숫자만 — LEV 접두사가 있는 경우에만 허용 (단독 숫자는 오인식 위험)
    """
    text_stripped = text.strip()

    # 1) LEV / LEC / LEU 계열
    # [: .;]* → 콜론/공백/점/세미콜론(; → : 오인식) 허용
    m = re.search(r"[Ll][Ee][VvCcUu][: .;_]*(\d+)", text_stripped)
    if m:
        return int(m.group(1))

    # 2) Lv.숫자 계열
    m = re.search(r"[Ll][Vv]\.?\s*(\d+)", text_stripped)
    if m:
        return int(m.group(1))

    # 3) 숫자만 — 단독 숫자는 무시 (42 같은 오인식 방지)
    #    LEV/Lv 접두사 없이 숫자만 나오면 다른 UI 요소일 가능성 높음
    # (필요 시 아래 주석 해제)
    # m = re.search(r"^\d+$", text_stripped.replace(" ", ""))
    # if m:
    #     val = int(m.group(0))
    #     if 1 <= val <= 99:
    #         return val

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
                 target_level: int = 5,
                 monitor: int = 1):
        self._level_roi    = level_roi
        self._hp_roi       = hp_roi
        self._target_level = target_level
        self._monitor_idx  = monitor

        # mss 직접 캡처 (ROI 좌표만 캡처 → 전체화면 캡처 불필요)
        self._sct          = _mss.mss()
        self._mon_left     = self._sct.monitors[monitor]["left"]
        self._mon_top      = self._sct.monitors[monitor]["top"]

        # easyocr lazy init
        self._ocr = None

        # 캐시
        self._last_level:    int   = None
        self._last_hp:       float = None
        self._last_level_t:  float = 0.0
        self._last_hp_t:     float = 0.0
        self._level_interval = 2.0   # 레벨: 2초마다 OCR
        self._hp_interval    = 0.5   # HP: 0.5초마다

        # 실패 로그 억제 (연속 실패 5회마다 1번만 출력)
        self._level_fail_count = 0
        self._level_fail_log_interval = 5

        print("[LevelDetector] 초기화 완료 (HP: 빨간색 HSV H=0~10+170~180, ROI 직접 캡처)")

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = _get_ocr()

    def update_rois(self, level_roi=None, hp_roi=None):
        if level_roi is not None:
            self._level_roi = level_roi
        if hp_roi is not None:
            self._hp_roi = hp_roi

    # ─────────────────────────────────────────────────────────
    #  ROI 직접 캡처 (mss로 해당 영역만 캡처)
    # ─────────────────────────────────────────────────────────

    def _grab_roi(self, roi: dict):
        """ROI 좌표만 mss로 직접 캡처 → BGR ndarray 반환.

        전체화면을 캡처 후 크롭하는 방식 대신
        해당 픽셀 영역만 캡처해 CPU/메모리 부하 최소화.
        """
        if roi is None:
            return None
        x = roi.get("x", 0)
        y = roi.get("y", 0)
        w = roi.get("w", roi.get("width", 0))
        h = roi.get("h", roi.get("height", 0))
        if w <= 0 or h <= 0:
            return None
        region = {
            "left":   self._mon_left + x,
            "top":    self._mon_top  + y,
            "width":  w,
            "height": h,
        }
        try:
            shot = self._sct.grab(region)
            bgr  = np.array(shot)[:, :, :3]   # BGRA → BGR
            return bgr
        except Exception:
            return None

    # 하위 호환 — frame_bgr 기반 크롭 (호출처가 넘겨줄 때만 사용)
    def _crop_roi(self, frame_bgr: np.ndarray, roi: dict):
        """ROI dict {"x","y","w","h"}로 frame_bgr 크롭 (레거시)."""
        if frame_bgr is None or roi is None:
            return None
        x = roi.get("x", 0)
        y = roi.get("y", 0)
        w = roi.get("w", roi.get("width", 0))
        h = roi.get("h", roi.get("height", 0))
        if w <= 0 or h <= 0:
            return None
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x+w), min(fh, y+h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    # ─────────────────────────────────────────────────────────
    #  레벨 인식
    # ─────────────────────────────────────────────────────────

    def read_level(self, frame_bgr: np.ndarray = None) -> int:
        """레벨 ROI를 mss로 직접 캡처 → OCR → 레벨 숫자 반환.

        frame_bgr: 하위 호환용 (무시됨 — mss 직접 캡처 사용)
        캐시: 2초
        """
        now = time.monotonic()
        if now - self._last_level_t < self._level_interval:
            return self._last_level
        self._last_level_t = now

        if self._level_roi is None:
            return self._last_level

        # 전체화면 대신 레벨 ROI 영역만 직접 캡처
        crop = self._grab_roi(self._level_roi)
        if crop is None:
            return self._last_level

        # 전처리 (2배 확대 + OTSU — 실측 최적)
        processed = _preprocess_for_ocr(crop)

        # OCR — 1회 실패 시 즉시 재시도 1회 (easyocr 비결정성 대응)
        def _run_ocr(img):
            try:
                self._ensure_ocr()
                return self._ocr.readtext(
                    img,
                    detail=1,
                    paragraph=False,
                    allowlist="LEVlev:;. 0123456789",  # 레벨 관련 문자만 허용
                    text_threshold=0.5,   # 글자 신뢰도 기준 상향 (기본 0.7→0.5로 완화)
                    low_text=0.3,         # 텍스트 영역 탐지 민감도
                )
            except Exception as e:
                print(f"[LevelDetector] easyocr 오류: {e}")
                return []

        def _try_parse(results):
            for (_, text, confidence) in results:
                if confidence < 0.1:
                    continue
                level = _parse_level(text)
                if level is not None and 1 <= level <= 99:
                    return level, text, confidence
            return None, None, None

        results = _run_ocr(processed)
        level, text, conf = _try_parse(results)

        # 1차 실패 → 즉시 재캡처 후 재시도
        if level is None:
            crop2 = self._grab_roi(self._level_roi)
            if crop2 is not None:
                processed2 = _preprocess_for_ocr(crop2)
                results2 = _run_ocr(processed2)
                level, text, conf = _try_parse(results2)
                if level is not None:
                    results = results2  # 로그용

        if level is not None:
            if level != self._last_level:
                print(f"[LevelDetector] 레벨: {self._last_level} → {level}"
                      f"  (OCR='{text}', conf={conf:.2f})")
            self._last_level = level
            self._level_fail_count = 0
            return level

        # 최종 실패 → 캐시 유지, 로그는 N회마다 1번만
        self._level_fail_count += 1
        if self._level_fail_count % self._level_fail_log_interval == 1:
            debug_list = [(t, f"{c:.2f}") for (_, t, c) in results]
            print(f"[LevelDetector] 레벨 인식 실패({self._level_fail_count}회). OCR={debug_list}")
        return self._last_level

    def is_target_level_reached(self, frame_bgr: np.ndarray) -> bool:
        lv = self.read_level(frame_bgr)
        if lv is None:
            return False
        return lv >= self._target_level

    # ─────────────────────────────────────────────────────────
    #  HP 인식
    # ─────────────────────────────────────────────────────────

    def read_hp(self, frame_bgr: np.ndarray = None) -> float:
        """HP ROI를 mss로 직접 캡처 → 파란색 픽셀 열 비율로 HP% 계산.

        frame_bgr: 하위 호환용 (무시됨 — mss 직접 캡처 사용)
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

        # 전체화면 대신 HP ROI 영역만 직접 캡처
        crop = self._grab_roi(self._hp_roi)
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
