"""
debug_ocr.py - level_roi 크롭 → 전처리 이미지 저장 + OCR 결과 출력
사용: python tools/debug_ocr.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, time
import cv2
import mss
import numpy as np

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

roi     = cfg["level_detector"]["level_roi"]
mon_idx = cfg.get("capture", {}).get("monitor", 1)

print(f"level_roi = {roi}")
print("3초 후 캡처 (게임 화면 준비)...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

with mss.mss() as sct:
    mon   = sct.monitors[mon_idx]
    raw   = sct.grab(mon)
    frame = np.array(raw)[:, :, :3]

x = roi["x"]; y = roi["y"]
w = roi.get("w", roi.get("width", 0))
h = roi.get("h", roi.get("height", 0))
crop = frame[y:y+h, x:x+w]

# ── 원본 크롭 저장
cv2.imwrite("ocr_1_crop_orig.png", crop)
print(f"[1] 원본 크롭 저장: ocr_1_crop_orig.png  shape={crop.shape}")

# ── 전처리 단계별 저장
enlarged = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("ocr_2_enlarged.png", enlarged)
print("[2] 3배 확대 저장: ocr_2_enlarged.png")

gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
cv2.imwrite("ocr_3_gray.png", gray)
print("[3] 그레이스케일 저장: ocr_3_gray.png")

clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
enhanced = clahe.apply(gray)
cv2.imwrite("ocr_4_clahe.png", enhanced)
print("[4] CLAHE 저장: ocr_4_clahe.png")

_, binary_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("ocr_5a_binary_INV.png", binary_inv)
print("[5a] THRESH_BINARY_INV 저장: ocr_5a_binary_INV.png  (구버전)")

_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
final = cv2.bitwise_not(binary)
cv2.imwrite("ocr_5b_binary_NOT.png", final)
print("[5b] THRESH_BINARY+bitwise_not 저장: ocr_5b_binary_NOT.png  (신버전)")

# ── OCR 결과 비교
import easyocr
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

res_inv = reader.readtext(binary_inv, detail=1, paragraph=False)
res_new = reader.readtext(final,      detail=1, paragraph=False)

print(f"\n[OCR 구버전 BINARY_INV] {[(t,f'{c:.2f}') for _,t,c in res_inv]}")
print(f"[OCR 신버전 BINARY+NOT] {[(t,f'{c:.2f}') for _,t,c in res_new]}")
