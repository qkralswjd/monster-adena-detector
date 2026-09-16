"""
debug_level_roi.py - 현재 화면에서 level_roi 크롭 이미지를 저장해 확인
사용: python tools/debug_level_roi.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, time
import cv2
import mss
import numpy as np

# config 로드
with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

roi = cfg["level_detector"]["level_roi"]
print(f"[debug] level_roi = {roi}")

mon_idx = cfg.get("capture", {}).get("monitor", 1)

with mss.mss() as sct:
    mon = sct.monitors[mon_idx]
    print(f"[debug] monitor = {mon}")

    # 3초 카운트다운 (사냥터 화면으로 전환할 시간)
    for i in range(3, 0, -1):
        print(f"[debug] {i}초 후 캡처...")
        time.sleep(1)

    # 전체화면 캡처
    raw = sct.grab(mon)
    frame = np.array(raw)[:, :, :3]   # BGR
    print(f"[debug] 전체화면 크기: {frame.shape}")  # (H, W, 3)

    x, y = roi["x"], roi["y"]
    w = roi.get("w", roi.get("width", 0))
    h = roi.get("h", roi.get("height", 0))

    # ROI 크롭
    crop = frame[y:y+h, x:x+w]
    print(f"[debug] crop 크기: {crop.shape}")

    # 저장
    cv2.imwrite("debug_level_crop.png", crop)
    print("[debug] debug_level_crop.png 저장 완료")

    # 전체화면에 ROI 박스 표시
    vis = frame.copy()
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # 1/4 축소 후 저장
    small = cv2.resize(vis, (vis.shape[1]//4, vis.shape[0]//4))
    cv2.imwrite("debug_level_fullscreen.png", small)
    print("[debug] debug_level_fullscreen.png 저장 완료 (1/4 축소)")

    print("\n[debug] 두 이미지를 열어서 ROI 위치가 맞는지 확인하세요.")
