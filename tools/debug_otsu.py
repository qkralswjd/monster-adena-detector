"""
debug_otsu.py - 같은 ROI를 5번 연속 캡처해서 OTSU 임계값과 전처리 결과 비교
사용: python tools/debug_otsu.py
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

x = roi["x"]; y = roi["y"]
w = roi.get("w", roi.get("width", 0))
h = roi.get("h", roi.get("height", 0))

print(f"level_roi = {roi}")
print("3초 후 5회 연속 캡처...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

with mss.mss() as sct:
    mon = sct.monitors[mon_idx]
    region = {"left": mon["left"]+x, "top": mon["top"]+y, "width": w, "height": h}

    for i in range(5):
        shot  = sct.grab(region)
        crop  = np.array(shot)[:, :, :3]

        # 전처리
        enlarged = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_LINEAR)
        gray     = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        thresh, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final    = cv2.bitwise_not(binary)

        cv2.imwrite(f"otsu_{i+1}_thresh{int(thresh)}.png", final)
        print(f"[{i+1}] OTSU 임계값={int(thresh)}  저장: otsu_{i+1}_thresh{int(thresh)}.png")
        time.sleep(0.3)

print("\n임계값이 매번 다르면 OTSU 불안정 → 고정 임계값으로 교체 필요")
