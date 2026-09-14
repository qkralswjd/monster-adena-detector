"""
test_click_pos.py
-----------------
실시간 화면 캡처창에서 클릭하면 피코가 해당 좌표로 이동+클릭.
"""

import sys
import time
import serial
import json
import os
import re
import cv2
import numpy as np
import mss
import threading
import ctypes


def send(ser, text):
    ser.write((text + "\n").encode("utf-8"))
    ser.flush()


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    port        = cfg["controller"]["port"]
    monitor_idx = cfg["capture"]["monitor"]

    # controller.py에서 SCALE 읽기
    ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
    with open(ctrl_path, encoding="utf-8") as f:
        code = f.read()
    scale_x = float(re.search(r"SCALE_X\s*=\s*([\d.]+)", code).group(1))
    scale_y = float(re.search(r"SCALE_Y\s*=\s*([\d.]+)", code).group(1))

    print("=" * 50)
    print("  실시간 클릭 테스트")
    print(f"  PORT={port}  SCALE_X={scale_x}  SCALE_Y={scale_y}")
    print("  캡처창에서 클릭 → 피코가 해당 위치 클릭")
    print("  ESC = 종료")
    print("=" * 50)

    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(0.5)
        print(f"✅ {port} 연결됨")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        sys.exit(1)

    # 모니터 정보
    with mss.mss() as sct:
        mon = sct.monitors[monitor_idx]
        mon_left = mon["left"]
        mon_top  = mon["top"]
        screen_w = mon["width"]
        screen_h = mon["height"]

    # 커서 리셋 - Windows + 피코 동시 (0,0) 동기화
    print("커서 리셋 중...")
    ctypes.windll.user32.SetCursorPos(0, 0)  # Windows 커서 (0,0)
    time.sleep(0.1)
    send(ser, "MOVE:-9999:-9999")             # 피코 커서 (0,0)
    time.sleep(1.0)
    cur_x, cur_y = 0, 0
    print(f"리셋 완료 → Windows + 피코 모두 (0,0) 동기화")
    print()

    # 클릭 이벤트 처리 (별도 스레드)
    click_queue = []
    lock = threading.Lock()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 디스플레이 좌표 → 원본 프레임 좌표
            fx = x * 2
            fy = y * 2
            # 전체화면 좌표
            sc_x = fx + mon_left
            sc_y = fy + mon_top
            with lock:
                click_queue.append((fx, fy, sc_x, sc_y))

    WIN = "실시간 클릭 테스트 (ESC=종료)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, screen_w // 2, screen_h // 2)
    cv2.setMouseCallback(WIN, on_mouse)

    sct = mss.mss()
    mon_rect = sct.monitors[monitor_idx]

    last_click_info = None  # 마지막 클릭 정보 (화면에 표시용)

    while True:
        # 실시간 캡처
        shot = sct.grab(mon_rect)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        disp = cv2.resize(frame, (screen_w // 2, screen_h // 2))

        # 클릭 처리
        with lock:
            if click_queue:
                fx, fy, sc_x, sc_y = click_queue.pop(0)

                # 피코 이동+클릭
                dx = sc_x - cur_x
                dy = sc_y - cur_y
                sdx = round(dx * scale_x)
                sdy = round(dy * scale_y)

                print(f"클릭 → 프레임({fx},{fy})  전체화면({sc_x},{sc_y})  "
                      f"MOVE({sdx},{sdy})")

                if sdx != 0 or sdy != 0:
                    send(ser, f"MOVE:{sdx}:{sdy}")
                    time.sleep(0.05)
                send(ser, "CLICK:80")

                cur_x, cur_y = sc_x, sc_y
                last_click_info = (fx // 2, fy // 2, sc_x, sc_y)

        # 마지막 클릭 위치 표시
        if last_click_info:
            dx2, dy2, scx2, scy2 = last_click_info
            cv2.circle(disp, (dx2, dy2), 10, (0, 0, 255), -1)
            cv2.circle(disp, (dx2, dy2), 10, (255, 255, 255), 2)
            cv2.putText(disp, f"({scx2},{scy2})", (dx2 + 12, dy2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    sct.close()
    ser.close()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
