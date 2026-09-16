"""
debug_preprocess.py - 전처리 방식별 OCR 결과 비교
사용: python tools/debug_preprocess.py

[테스트 결과 요약 — 2025-09]
  확대(resize) 후 OTSU 이진화를 하면 보간 픽셀이 '1'의 얇은 세로획을 채워
  '9' 또는 '4'처럼 변형됨 → 오인식 원인.

  ✅ 승자: 원본 크기 그대로 OTSU 이진화 (resize 없음)
     → 이진화 결과(흰배경+검정텍스트)를 easyocr에 직접 입력
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, time
import cv2
import mss
import numpy as np
import easyocr

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

roi     = cfg["level_detector"]["level_roi"]
mon_idx = cfg.get("capture", {}).get("monitor", 1)
x = roi["x"]; y = roi["y"]
w = roi.get("w", roi.get("width", 0))
h = roi.get("h", roi.get("height", 0))

print("3초 후 캡처...")
for i in range(3, 0, -1):
    print(f"  {i}..."); time.sleep(1)

with mss.mss() as sct:
    mon    = sct.monitors[mon_idx]
    region = {"left": mon["left"]+x, "top": mon["top"]+y, "width": w, "height": h}
    shot   = sct.grab(region)
    crop   = np.array(shot)[:, :, :3]

cv2.imwrite("pre_0_orig.png", crop)
print(f"원본 크기: {crop.shape}")

reader = easyocr.Reader(["en"], gpu=False, verbose=False)
ALLOW  = "LEVlev:;. 0123456789"

def ocr(img, name):
    res = reader.readtext(img, detail=1, paragraph=False,
                          allowlist=ALLOW, text_threshold=0.5, low_text=0.3)
    parsed = [(t, f"{c:.2f}") for _,t,c in res]
    ok = "✅" if any("15" in t or "1" in t for t,_ in parsed) else "❌"
    print(f"  {ok} [{name}] → {parsed}")
    return img

# ── ★ NEW (현재 코드): OTSU만 — resize 없음
def preprocess_NEW(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(b)

# ── A (구 코드): 3배 확대 + CLAHE + OTSU
def preprocess_A(crop):
    h, w = crop.shape[:2]
    large = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_LINEAR)
    gray  = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enh   = clahe.apply(gray)
    _, b  = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(b)

# ── B: 2배 INTER_CUBIC + OTSU
def preprocess_B(crop):
    h, w = crop.shape[:2]
    large = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    _, b  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(b)

# ── C: 4배 INTER_CUBIC + 고정 임계값 147
def preprocess_C(crop):
    h, w = crop.shape[:2]
    large = cv2.resize(crop, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    _, b  = cv2.threshold(gray, 147, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(b)

# ── D: 3배 INTER_CUBIC + 고정 thresh200 (흰 텍스트만)
def preprocess_D(crop):
    h, w = crop.shape[:2]
    large = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    _, b  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(b)

# ── E: 3x 원본 컬러
def preprocess_E(crop):
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)

print("\n=== 전처리 방식별 OCR 결과 ===")
print("  (★ = 현재 코드에 적용된 방식)\n")
for name, fn in [("★NEW_OTSU만(resize없음)", preprocess_NEW),
                 ("A_3x+CLAHE+OTSU(구버전)", preprocess_A),
                 ("B_2x+OTSU",               preprocess_B),
                 ("C_4x+고정thresh147",       preprocess_C),
                 ("D_3x+thresh200",           preprocess_D),
                 ("E_3x+원본컬러",            preprocess_E)]:
    img = fn(crop)
    cv2.imwrite(f"pre_{name[:4]}.png", img)
    ocr(img, name)
