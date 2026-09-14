"""
main.py - 몬스터 탐지 + 좌표 로그
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cap = ScreenCapture(cfg["capture"]["monitor"])
    mon = cap._monitor
    print(f"[Capture] monitor left={mon['left']} top={mon['top']} {mon['width']}x{mon['height']}")

    d = cfg["detector"]
    det = YOLODetector(
        model_path    = d["model"],
        confidence    = d["confidence"],
        iou_threshold = d["iou_threshold"],
        device        = d["device"],
        img_size      = d["img_size"],
    )

    r = cfg["roi"]
    roi = (r["x"], r["y"], r["width"], r["height"]) if r["width"] > 0 else None
    print(f"[ROI] {roi}")
    print("Ctrl+C 종료\n")

    try:
        while True:
            frame = cap.capture()
            if frame is None:
                continue

            dets = det.detect(frame)

            if roi:
                rx, ry, rw, rh = roi
                dets = [d for d in dets if rx <= d.cx <= rx+rw and ry <= d.cy <= ry+rh]

            monsters = [d for d in dets if d.class_id == 0]

            for i, m in enumerate(monsters, 1):
                print(f"[몬스터 {i}] 프레임({m.cx}, {m.cy})  bbox=({m.x},{m.y},{m.w},{m.h})  conf={m.confidence:.2f}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("종료")
