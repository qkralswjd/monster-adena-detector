"""
set_roi.py
----------
게임 화면을 캡처한 뒤 마우스로 드래그해서 ROI(탐지 영역)를 설정.
설정된 ROI는 config.json에 자동 저장됨.

실행:
    python set_roi.py

조작:
    마우스 드래그 : ROI 영역 선택
    Enter / Space : 확정 저장
    R             : 다시 그리기
    ESC           : 취소 (저장 안 함)
"""

import cv2
import mss
import numpy as np
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_roi(x, y, w, h):
    cfg = load_config()
    cfg["roi"]["x"] = x
    cfg["roi"]["y"] = y
    cfg["roi"]["width"] = w
    cfg["roi"]["height"] = h
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"[ROI] config.json 저장 완료: x={x}, y={y}, w={w}, h={h}")


def capture_screen(monitor_idx):
    with mss.MSS() as sct:
        mon = sct.monitors[monitor_idx]
        shot = sct.grab(mon)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame, mon


def main():
    cfg = load_config()
    monitor_idx = cfg["capture"]["monitor"]
    current_roi = cfg["roi"]

    print(f"[ROI] 게임 모니터 인덱스: {monitor_idx}")
    print(f"[ROI] 현재 ROI: x={current_roi['x']}, y={current_roi['y']}, "
          f"w={current_roi['width']}, h={current_roi['height']}")
    print()
    print("[ROI] 조작법:")
    print("  마우스 드래그 : ROI 영역 선택")
    print("  Enter / Space : 확정 저장")
    print("  R             : 다시 그리기")
    print("  ESC           : 취소")
    print()

    # 게임 화면 캡처
    print("[ROI] 화면 캡처 중...")
    frame, mon = capture_screen(monitor_idx)
    h, w = frame.shape[:2]
    print(f"[ROI] 캡처 크기: {w}x{h}")

    # 표시용 축소 (너무 크면 화면에 안 맞음)
    MAX_W, MAX_H = 1280, 720
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    display = cv2.resize(frame, (disp_w, disp_h))

    # 기존 ROI 표시
    if current_roi["width"] > 0 and current_roi["height"] > 0:
        rx = int(current_roi["x"] * scale)
        ry = int(current_roi["y"] * scale)
        rw = int(current_roi["width"] * scale)
        rh = int(current_roi["height"] * scale)
        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(display, "현재 ROI", (rx, ry - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    WIN = "ROI 설정 (드래그→Enter저장 / R재시도 / ESC취소)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    roi_data = {"start": None, "end": None, "drawing": False, "done": False}
    base = display.copy()

    def mouse_cb(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_data["start"] = (mx, my)
            roi_data["end"]   = (mx, my)
            roi_data["drawing"] = True
            roi_data["done"]  = False
        elif event == cv2.EVENT_MOUSEMOVE and roi_data["drawing"]:
            roi_data["end"] = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_data["end"] = (mx, my)
            roi_data["drawing"] = False
            roi_data["done"] = True

    cv2.setMouseCallback(WIN, mouse_cb)

    while True:
        img = base.copy()

        # 드래그 중 / 완료 사각형 표시
        if roi_data["start"] and roi_data["end"]:
            sx, sy = roi_data["start"]
            ex, ey = roi_data["end"]
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            color = (0, 200, 255) if roi_data["drawing"] else (0, 80, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 실제 픽셀 좌표 표시
            real_x1 = int(x1 / scale)
            real_y1 = int(y1 / scale)
            real_w  = int((x2 - x1) / scale)
            real_h  = int((y2 - y1) / scale)
            label = f"x={real_x1} y={real_y1} w={real_w} h={real_h}"
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if roi_data["done"]:
                cv2.putText(img, "Enter=저장  R=재시도  ESC=취소",
                            (10, disp_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 안내 텍스트
        if not roi_data["start"]:
            cv2.putText(img, "드래그로 탐지 영역을 선택하세요",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF

        # Enter / Space → 저장
        if key in (13, 32) and roi_data["done"]:
            sx, sy = roi_data["start"]
            ex, ey = roi_data["end"]
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            real_x = int(x1 / scale)
            real_y = int(y1 / scale)
            real_w = int((x2 - x1) / scale)
            real_h = int((y2 - y1) / scale)
            if real_w > 10 and real_h > 10:
                save_roi(real_x, real_y, real_w, real_h)
                print(f"[ROI] ✅ 저장 완료!")
                print(f"      탐지 영역: ({real_x},{real_y}) {real_w}x{real_h}")
                print(f"      이제 python main.py 실행하세요.")
            else:
                print("[ROI] 너무 작은 영역입니다. 다시 드래그하세요.")
            break

        # R → 초기화
        elif key == ord("r") or key == ord("R"):
            roi_data = {"start": None, "end": None, "drawing": False, "done": False}
            base = display.copy()
            print("[ROI] 초기화. 다시 드래그하세요.")

        # ESC → 취소
        elif key == 27:
            print("[ROI] 취소. config.json 변경 없음.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
