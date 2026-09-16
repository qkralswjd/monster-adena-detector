"""
capture_tool.py
---------------
학습 데이터 캡처 도구.

사용법:
  python capture_tool.py

단축키:
  T     : 현재 화면 캡처 저장
  Q     : 종료

저장 위치:
  dataset/images/train/capture_YYYYMMDD_HHMMSS_NNN.png
"""

import os
import time
import threading
from datetime import datetime

import mss
import cv2
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "dataset", "images", "train")
os.makedirs(SAVE_DIR, exist_ok=True)

capture_count = 0


def capture_screen():
    global capture_count
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_count += 1
    filename = f"capture_{ts}_{capture_count:03d}.png"
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"[캡처] 저장: {filename}  (총 {capture_count}장)")


def main():
    print("=" * 40)
    print("  학습 데이터 캡처 도구")
    print("=" * 40)
    print(f"  저장 위치: {SAVE_DIR}")
    print(f"  T 키 : 캡처")
    print(f"  Q 키 : 종료")
    print("=" * 40)
    print("게임 창 띄우고 T 누르세요\n")

    import keyboard
    keyboard.add_hotkey("t", capture_screen)

    while True:
        if keyboard.is_pressed("q"):
            print(f"\n[종료] 총 {capture_count}장 캡처됨")
            print(f"저장 위치: {SAVE_DIR}")
            print("labelimg 로 라벨링 후 재학습 하세요")
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main()
