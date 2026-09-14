"""
capture_dataset.py
------------------
게임 화면을 일정 간격으로 자동 캡처해서 데이터셋 이미지 저장.

사용법:
    python tools/capture_dataset.py

조작:
    S     : 즉시 캡처 (수동)
    Space : 자동 캡처 ON/OFF 토글
    R     : ROI 재설정
    Q/ESC : 종료

저장 위치:
    dataset/images/raw/  ← 수집된 원본 이미지
"""

import cv2
import mss
import numpy as np
import time
import os
import sys
from datetime import datetime

# ── 설정 ────────────────────────────────────────
SAVE_DIR      = os.path.join(os.path.dirname(__file__), "..", "dataset", "images", "raw")
AUTO_INTERVAL = 1.5    # 자동 캡처 간격 (초) - 게임 속도에 맞게 조절
MONITOR_IDX   = 1      # 모니터 번호 (1=기본)
# ────────────────────────────────────────────────


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def select_roi(frame):
    """마우스 드래그로 ROI 선택."""
    clone = frame.copy()
    roi_data = {"start": None, "end": None, "done": False}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_data["start"] = (x, y)
            roi_data["end"]   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and roi_data["start"]:
            roi_data["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_data["end"]  = (x, y)
            roi_data["done"] = True

    win = "ROI 선택 (드래그 → Enter 확정 / ESC 취소)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        disp = clone.copy()
        if roi_data["start"] and roi_data["end"]:
            cv2.rectangle(disp, roi_data["start"], roi_data["end"], (0, 255, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and roi_data["done"]:   # Enter or Space
            break
        if key == 27:
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    s, e = roi_data["start"], roi_data["end"]
    if s and e:
        x = min(s[0], e[0]); y = min(s[1], e[1])
        w = abs(e[0] - s[0]); h = abs(e[1] - s[1])
        if w > 10 and h > 10:
            return (x, y, w, h)
    return None


def main():
    ensure_dir(SAVE_DIR)

    sct     = mss.mss()
    monitor = sct.monitors[MONITOR_IDX]
    roi     = None   # None이면 전체 화면

    # 초기 화면 캡처해서 ROI 선택
    shot  = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    print("[캡처] ROI를 드래그로 선택하세요 (게임 화면 영역)")
    result = select_roi(frame)
    if result:
        roi = {"top": result[1], "left": result[0],
               "width": result[2], "height": result[3]}
        print(f"[캡처] ROI 설정: {result}")
    else:
        print("[캡처] ROI 없음 → 전체 화면 캡처")

    count       = 0
    auto_on     = False
    last_auto   = 0.0

    WIN = "Dataset Capture  |  Space=자동ON/OFF  S=수동  R=ROI  Q=종료"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print(f"\n[캡처] 저장 경로: {os.path.abspath(SAVE_DIR)}")
    print("[캡처] Space=자동캡처토글  S=수동캡처  R=ROI재설정  Q=종료\n")

    while True:
        region = roi if roi else monitor
        shot   = sct.grab(region)
        frame  = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

        # ── 자동 캡처 ───────────────────────────
        now = time.time()
        if auto_on and (now - last_auto) >= AUTO_INTERVAL:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.join(SAVE_DIR, f"{ts}.jpg")
            cv2.imwrite(path, frame)
            count    += 1
            last_auto = now
            print(f"  [자동] {count:04d} {os.path.basename(path)}")

        # ── HUD ─────────────────────────────────
        disp   = frame.copy()
        status = "AUTO ON " if auto_on else "AUTO OFF"
        color  = (0, 255, 0) if auto_on else (0, 0, 255)
        cv2.putText(disp, f"{status}  |  saved: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(disp, f"interval: {AUTO_INTERVAL}s",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(WIN, disp)

        # ── 키 입력 ─────────────────────────────
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord(' '):          # Space: 자동 ON/OFF
            auto_on = not auto_on
            print(f"[캡처] 자동 캡처 {'ON' if auto_on else 'OFF'}")

        elif key == ord('s'):          # S: 수동 캡처
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.join(SAVE_DIR, f"{ts}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  [수동] {count:04d} {os.path.basename(path)}")

        elif key == ord('r'):          # R: ROI 재설정
            full = sct.grab(monitor)
            full_frame = cv2.cvtColor(np.array(full), cv2.COLOR_BGRA2BGR)
            result = select_roi(full_frame)
            if result:
                roi = {"top": result[1], "left": result[0],
                       "width": result[2], "height": result[3]}
                print(f"[캡처] ROI 재설정: {result}")
            else:
                roi = None
                print("[캡처] ROI 해제 → 전체 화면")

    cv2.destroyAllWindows()
    print(f"\n[캡처] 완료. 총 {count}장 저장 → {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
