"""
set_roi.py
----------
마우스 드래그로 ROI(탐지 영역)를 직접 설정하는 도구.

사용법:
  python set_roi.py

  1. 화면 캡처가 뜸
  2. 마우스로 탐지할 영역을 드래그
  3. 엔터 or S키 → config.json에 자동 저장
  4. R키 → 다시 그리기
  5. Q/ESC → 취소
"""

import cv2
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from screen_capture import ScreenCapture

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# ── 전역 상태 ──────────────────────────────────
drawing   = False
ix, iy    = -1, -1
ex, ey    = -1, -1
rect_done = False
frame_orig = None


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, ex, ey, rect_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing   = True
        rect_done = False
        ix, iy = x, y
        ex, ey = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing   = False
        rect_done = True
        ex, ey = x, y


def get_roi_rect():
    """드래그 좌표 → (x, y, w, h) 정규화."""
    x1 = min(ix, ex)
    y1 = min(iy, ey)
    x2 = max(ix, ex)
    y2 = max(iy, ey)
    return x1, y1, x2 - x1, y2 - y1


def save_roi(x, y, w, h):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["roi"] = {"x": x, "y": y, "width": w, "height": h}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"[ROI] 저장 완료: x={x}, y={y}, w={w}, h={h}")


def main():
    global frame_orig

    # 현재 화면 캡처
    cap = ScreenCapture(1)
    frame_orig = cap.capture()
    if frame_orig is None:
        print("[ROI] 캡처 실패")
        return

    # 현재 config ROI 읽기
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cur = cfg.get("roi", {})
    cur_x, cur_y = cur.get("x", 0), cur.get("y", 0)
    cur_w, cur_h = cur.get("width", 0), cur.get("height", 0)

    # 표시용으로 절반 크기로 리사이즈 (1920x1080 → 960x540)
    SCALE = 0.5
    display = cv2.resize(frame_orig, (0, 0), fx=SCALE, fy=SCALE)
    h_d, w_d = display.shape[:2]

    WIN = "ROI 설정 - 드래그로 탐지영역 선택 | S=저장  R=초기화  Q=취소"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, w_d, h_d)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("=" * 55)
    print(" ROI 설정 도구")
    print("=" * 55)
    print(" 마우스 드래그로 탐지할 영역을 선택하세요")
    print(" S / Enter : 저장")
    print(" R         : 다시 그리기")
    print(" Q / ESC   : 취소")
    print()
    if cur_w > 0:
        print(f" 현재 ROI: x={cur_x} y={cur_y} w={cur_w} h={cur_h}")
    else:
        print(" 현재 ROI: 없음 (전체화면 탐지 중)")
    print()

    while True:
        img = display.copy()

        # 현재 ROI 표시 (초록 점선)
        if cur_w > 0:
            x1s = int(cur_x * SCALE)
            y1s = int(cur_y * SCALE)
            x2s = int((cur_x + cur_w) * SCALE)
            y2s = int((cur_y + cur_h) * SCALE)
            # 점선 효과 (10px 간격)
            for i in range(x1s, x2s, 20):
                cv2.line(img, (i, y1s), (min(i+10, x2s), y1s), (0, 255, 0), 1)
                cv2.line(img, (i, y2s), (min(i+10, x2s), y2s), (0, 255, 0), 1)
            for i in range(y1s, y2s, 20):
                cv2.line(img, (x1s, i), (x1s, min(i+10, y2s)), (0, 255, 0), 1)
                cv2.line(img, (x2s, i), (x2s, min(i+10, y2s)), (0, 255, 0), 1)
            cv2.putText(img, f"현재ROI({cur_x},{cur_y} {cur_w}x{cur_h})",
                        (x1s+4, y1s+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 드래그 중인 사각형 (파란색)
        if ix >= 0 and iy >= 0:
            cv2.rectangle(img, (ix, iy), (ex, ey), (255, 100, 0), 2)

            # 크기 표시 (실제 픽셀 기준)
            rw = int(abs(ex - ix) / SCALE)
            rh = int(abs(ey - iy) / SCALE)
            rx = int(min(ix, ex) / SCALE)
            ry = int(min(iy, ey) / SCALE)
            cv2.putText(img,
                        f"x={rx} y={ry}  {rw}x{rh}px",
                        (min(ix, ex), max(min(iy, ey) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

            if rect_done:
                cv2.putText(img, "S=저장  R=다시그리기",
                            (10, h_d - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # 안내 텍스트
        cv2.putText(img, "드래그로 탐지영역 선택 | S=저장 R=초기화 Q=취소",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(WIN, img)
        key = cv2.waitKey(16) & 0xFF

        if key in (ord('q'), 27):   # Q / ESC
            print("[ROI] 취소")
            break

        elif key in (ord('s'), 13):  # S / Enter
            if rect_done and abs(ex - ix) > 10 and abs(ey - iy) > 10:
                rx, ry, rw, rh = get_roi_rect()
                # 실제 픽셀로 변환
                real_x = int(rx / SCALE)
                real_y = int(ry / SCALE)
                real_w = int(rw / SCALE)
                real_h = int(rh / SCALE)
                save_roi(real_x, real_y, real_w, real_h)
                print(f"[ROI] 적용됨: x={real_x} y={real_y} w={real_w} h={real_h}")
                print(f"[ROI] main.py 재실행하면 적용됩니다")
                break
            else:
                print("[ROI] 먼저 영역을 드래그로 선택하세요")

        elif key == ord('r'):        # R : 초기화
            ix, iy, ex, ey = -1, -1, -1, -1
            rect_done = False
            print("[ROI] 초기화")

        elif key == ord('0'):        # 0 : ROI 해제 (전체화면)
            save_roi(0, 0, 0, 0)
            print("[ROI] 전체화면 탐지로 초기화됨")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
