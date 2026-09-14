"""
capture_dataset.py
------------------
게임 실행 중 T키로 전체화면 캡처해서 dataset/images/raw/ 에 저장.
main.py 실행 중에도 쓸 수 있도록 단독 실행 가능.

조작:
    T     : 현재 화면 캡처 저장
    Q/ESC : 종료

저장 위치:
    dataset/images/raw/
"""

import cv2
import mss
import numpy as np
import time
import os
from datetime import datetime

# ── 설정 ────────────────────────────────────────
SAVE_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset", "images", "raw")
MONITOR_IDX = 1
# ────────────────────────────────────────────────


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    sct     = mss.mss()
    monitor = sct.monitors[MONITOR_IDX]
    count   = 0

    WIN = "Capture  |  T=캡처저장  Q=종료"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print(f"[캡처] 저장 경로: {os.path.abspath(SAVE_DIR)}")
    print("[캡처] T=캡처  Q=종료\n")

    while True:
        shot  = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

        # HUD
        disp = frame.copy()
        cv2.putText(disp, f"T=캡처  saved:{count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in (ord('t'), ord('T')):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.join(SAVE_DIR, f"{ts}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  [{count:04d}] {os.path.basename(path)}")

    cv2.destroyAllWindows()
    print(f"\n[캡처] 완료. 총 {count}장 → {os.path.abspath(SAVE_DIR)}")
    print("  → 다음: python tools/label_tool.py")


if __name__ == "__main__":
    main()
