"""
test_click_pos.py
-----------------
화면 캡처 후 클릭한 위치로 피코가 이동 + 클릭.
실제 클릭 좌표가 맞는지 확인용.
"""

import sys
import time
import serial
import json
import os
import cv2
import numpy as np
import mss


def send(ser, text):
    ser.write((text + "\n").encode("utf-8"))
    ser.flush()
    time.sleep(0.05)


def capture_screen(monitor_idx=1):
    with mss.mss() as sct:
        mon = sct.monitors[monitor_idx]
        shot = sct.grab(mon)
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), mon


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    port        = cfg["controller"]["port"]
    monitor_idx = cfg["capture"]["monitor"]
    scale_x     = 0.395
    scale_y     = 0.395

    # controller.py에서 현재 SCALE 읽기
    ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
    import re
    with open(ctrl_path, encoding="utf-8") as f:
        code = f.read()
    m = re.search(r"SCALE_X\s*=\s*([\d.]+)", code)
    if m: scale_x = float(m.group(1))
    m = re.search(r"SCALE_Y\s*=\s*([\d.]+)", code)
    if m: scale_y = float(m.group(1))

    print("=" * 50)
    print("  피코 클릭 위치 테스트")
    print(f"  SCALE_X={scale_x}  SCALE_Y={scale_y}")
    print("=" * 50)

    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(0.5)
        print(f"✅ {port} 연결됨")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        sys.exit(1)

    # 커서 리셋
    print("커서 리셋 중...")
    send(ser, "MOVE:-9999:-9999")
    time.sleep(1.0)

    with mss.mss() as sct:
        mon = sct.monitors[monitor_idx]
        mon_left = mon["left"]
        mon_top  = mon["top"]
        screen_w = mon["width"]
        screen_h = mon["height"]

    # 리셋 후 게임 중앙으로
    cx = mon_left + screen_w // 2
    cy = mon_top  + screen_h // 2
    sx = round(cx * scale_x)
    sy = round(cy * scale_y)
    send(ser, f"MOVE:{sx}:{sy}")
    time.sleep(0.5)
    cur_x, cur_y = cx, cy
    print(f"커서 리셋 완료 → 전체화면 중앙({cx},{cy})")

    while True:
        print()
        print("─" * 50)
        print("화면 캡처 중...")
        frame, mon = capture_screen(monitor_idx)
        h, w = frame.shape[:2]
        disp = cv2.resize(frame, (w // 2, h // 2))

        result = {"pt": None}

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 원본 해상도 프레임 좌표
                fx = x * 2
                fy = y * 2
                # 전체화면 좌표
                sc_x = fx + mon_left
                sc_y = fy + mon_top
                result["pt"] = (fx, fy, sc_x, sc_y)

                cv2.circle(disp, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(disp, f"프레임({fx},{fy})", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(disp, f"전체화면({sc_x},{sc_y})", (x+10, y+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                cv2.imshow("클릭할 위치 선택 (ESC=종료)", disp)

        cv2.namedWindow("클릭할 위치 선택 (ESC=종료)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("클릭할 위치 선택 (ESC=종료)", w // 2, h // 2)
        cv2.setMouseCallback("클릭할 위치 선택 (ESC=종료)", on_click)
        cv2.imshow("클릭할 위치 선택 (ESC=종료)", disp)
        print("클릭할 위치를 선택하세요. (ESC=종료)")

        while result["pt"] is None:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                ser.close()
                print("종료")
                return

        cv2.destroyAllWindows()

        fx, fy, sc_x, sc_y = result["pt"]
        print(f"선택: 프레임({fx},{fy})  전체화면({sc_x},{sc_y})")

        # 피코 이동량 계산 (전체화면 기준)
        dx = sc_x - cur_x
        dy = sc_y - cur_y
        sdx = round(dx * scale_x)
        sdy = round(dy * scale_y)

        print(f"현재커서({cur_x},{cur_y}) → 목표({sc_x},{sc_y})")
        print(f"이동: dx={dx}, dy={dy} → MOVE:{sdx}:{sdy}")
        print(f"CLICK 전송...")

        if sdx != 0 or sdy != 0:
            send(ser, f"MOVE:{sdx}:{sdy}")
            time.sleep(0.1)
        send(ser, "CLICK:80")

        cur_x, cur_y = sc_x, sc_y
        print(f"✅ 클릭 완료!")
        print("실제로 해당 위치가 클릭됐는지 확인하세요.")


if __name__ == "__main__":
    main()
