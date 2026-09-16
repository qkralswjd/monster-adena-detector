"""
debug_rois.py - level_roi + hp_roi 직접 캡처 이미지 저장
사용: python tools/debug_rois.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, time
import cv2
import mss
import numpy as np

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

level_roi = cfg["level_detector"]["level_roi"]
hp_roi    = cfg["level_detector"]["hp_roi"]
mon_idx   = cfg.get("capture", {}).get("monitor", 1)

print(f"level_roi = {level_roi}")
print(f"hp_roi    = {hp_roi}")
print("3초 후 캡처...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

with mss.mss() as sct:
    mon = sct.monitors[mon_idx]
    print(f"monitor   = left={mon['left']} top={mon['top']} {mon['width']}x{mon['height']}")

    # ── 전체화면 캡처 후 크롭 (구버전 방식)
    full_raw  = sct.grab(mon)
    full_bgr  = np.array(full_raw)[:, :, :3]

    lx = level_roi["x"]; ly = level_roi["y"]
    lw = level_roi.get("w", level_roi.get("width", 0))
    lh = level_roi.get("h", level_roi.get("height", 0))
    level_crop_old = full_bgr[ly:ly+lh, lx:lx+lw]

    hx = hp_roi["x"]; hy = hp_roi["y"]
    hw = hp_roi.get("w", hp_roi.get("width", 0))
    hh = hp_roi.get("h", hp_roi.get("height", 0))
    hp_crop_old = full_bgr[hy:hy+hh, hx:hx+hw]

    # ── ROI 직접 캡처 (신버전 방식)
    level_region = {"left": mon["left"]+lx, "top": mon["top"]+ly, "width": lw, "height": lh}
    hp_region    = {"left": mon["left"]+hx, "top": mon["top"]+hy, "width": hw, "height": hh}

    level_crop_new = np.array(sct.grab(level_region))[:, :, :3]
    hp_crop_new    = np.array(sct.grab(hp_region))[:, :, :3]

# 저장
cv2.imwrite("roi_level_OLD.png", level_crop_old)
cv2.imwrite("roi_level_NEW.png", level_crop_new)
cv2.imwrite("roi_hp_OLD.png",    hp_crop_old)
cv2.imwrite("roi_hp_NEW.png",    hp_crop_new)

print("\n저장 완료:")
print("  roi_level_OLD.png  ← 전체화면 크롭 (구버전)")
print("  roi_level_NEW.png  ← 직접 캡처 (신버전)")
print("  roi_hp_OLD.png     ← 전체화면 크롭 (구버전)")
print("  roi_hp_NEW.png     ← 직접 캡처 (신버전)")
print("\nOLD/NEW 가 동일하면 좌표 정상, 다르면 오프셋 문제")
