"""
set_roi.py
----------
마우스 드래그로 ROI 영역 선택 후 config.json 에 자동 저장.

사용:
  python tools/set_roi.py

조작:
  마우스 드래그 : ROI 영역 선택
  Enter / Space : 확정 저장
  R             : 다시 선택
  Q / ESC       : 취소 (저장 안함)
"""

import cv2
import mss
import numpy as np
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# 드래그 상태
drag_start = None
drag_end   = None
dragging   = False


def mouse_cb(event, x, y, flags, param):
    global drag_start, drag_end, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        drag_end   = (x, y)
        dragging   = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_end = (x, y)
        dragging = False


def main():
    global drag_start, drag_end, dragging

    # 현재 화면 캡처
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw     = sct.grab(monitor)
        screen  = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

    # 화면 절반 크기로 표시 (4K 등 대응)
    sh, sw = screen.shape[:2]
    scale  = min(1.0, 1280 / sw)
    disp_w = int(sw * scale)
    disp_h = int(sh * scale)

    WIN = "ROI 설정  |  드래그=선택  Enter=저장  R=재선택  Q=취소"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("[ROI] 게임 화면에서 탐지할 영역을 드래그로 선택하세요")
    print("[ROI] Enter=저장  R=재선택  Q=취소")

    confirmed = False

    while True:
        display = screen.copy()

        # 드래그 중 or 완료 박스 표시
        if drag_start and drag_end:
            x1 = min(drag_start[0], drag_end[0])
            y1 = min(drag_start[1], drag_end[1])
            x2 = max(drag_start[0], drag_end[0])
            y2 = max(drag_start[1], drag_end[1])

            # 어두운 오버레이
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (sw, sh), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

            # 선택 영역은 밝게
            display[y1:y2, x1:x2] = screen[y1:y2, x1:x2]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            w = x2 - x1
            h = y2 - y1
            cv2.putText(display, f"{w}x{h}  ({x1},{y1})",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display, "Enter=저장  R=재선택  Q=취소",
                    (10, sh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if drag_start and drag_end:
                x1 = min(drag_start[0], drag_end[0])
                y1 = min(drag_start[1], drag_end[1])
                x2 = max(drag_start[0], drag_end[0])
                y2 = max(drag_start[1], drag_end[1])
                w  = x2 - x1
                h  = y2 - y1
                if w > 10 and h > 10:
                    confirmed = True
                    break
            print("[ROI] 영역을 먼저 드래그로 선택하세요")

        elif key == ord('r'):
            drag_start = None
            drag_end   = None
            print("[ROI] 다시 선택하세요")

        elif key in (ord('q'), 27):
            print("[ROI] 취소")
            break

    cv2.destroyAllWindows()

    if confirmed:
        x1 = min(drag_start[0], drag_end[0])
        y1 = min(drag_start[1], drag_end[1])
        x2 = max(drag_start[0], drag_end[0])
        y2 = max(drag_start[1], drag_end[1])
        w  = x2 - x1
        h  = y2 - y1

        # config.json 업데이트
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

        config["roi"] = {
            "enabled": True,
            "x": x1,
            "y": y1,
            "width":  w,
            "height": h,
            "_note": "enabled=true 면 해당 영역만 캡처 → FPS 향상. set_roi.py 로 설정."
        }

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n[ROI] 저장 완료!")
        print(f"  x={x1}  y={y1}  width={w}  height={h}")
        print(f"  → config.json 업데이트됨")
        print(f"\npython main.py 실행하면 ROI 적용됩니다")


if __name__ == "__main__":
    main()
