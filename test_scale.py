"""
test_scale.py
-------------
피코 HID SCALE 측정 도구 (화면 캡처 클릭 방식)

사용법:
  1. python test_scale.py
  2. 화면 캡처창 뜸
  3. 현재 커서 위치 클릭 (클릭 전 위치 저장)
  4. 자동으로 MOVE:100:0 전송
  5. 캡처창 다시 뜸 → 커서 이동된 위치 클릭 (클릭 후 위치 저장)
  6. 차이 계산 → SCALE 자동 출력 및 적용
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


def send(ser, text):
    ser.write((text + "\n").encode("utf-8"))
    ser.flush()
    time.sleep(0.1)


def capture_screen(monitor_idx=1):
    with mss.mss() as sct:
        mon = sct.monitors[monitor_idx]
        shot = sct.grab(mon)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def pick_point(frame, title="클릭하세요"):
    """화면 캡처 이미지에서 마우스 클릭 위치 반환"""
    result = {"pt": None}

    # 화면에 맞게 축소 (1920x1080 → 960x540)
    h, w = frame.shape[:2]
    disp = cv2.resize(frame, (w // 2, h // 2))

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result["pt"] = (x * 2, y * 2)  # 원본 해상도로 변환
            # 클릭 위치 표시
            cv2.circle(disp, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(disp, f"({x*2}, {y*2})", (x+10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(title, disp)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, w // 2, h // 2)
    cv2.setMouseCallback(title, on_click)
    cv2.imshow(title, disp)

    print(f"  → 창에서 해당 위치를 클릭하세요. (클릭 후 자동 진행)")
    while result["pt"] is None:
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    time.sleep(0.5)
    cv2.destroyWindow(title)
    return result["pt"]


def measure_axis(ser, axis_label, move_cmd, monitor_idx):
    """한 축 측정: 클릭 전 → MOVE → 클릭 후 → 차이 계산"""
    print(f"\n{'─'*50}")
    print(f"【{axis_label} 측정】 {move_cmd}")
    print(f"  1단계: 현재 커서 위치를 클릭하세요")

    frame1 = capture_screen(monitor_idx)
    pt1 = pick_point(frame1, f"{axis_label} - 이동 전 커서 위치 클릭")
    if pt1 is None:
        print("  ❌ 취소됨")
        return None

    print(f"  이동 전: {pt1}")
    print(f"  {move_cmd} 전송 중...")
    send(ser, move_cmd)
    time.sleep(0.3)

    print(f"  2단계: 이동된 커서 위치를 클릭하세요")
    frame2 = capture_screen(monitor_idx)
    pt2 = pick_point(frame2, f"{axis_label} - 이동 후 커서 위치 클릭")
    if pt2 is None:
        print("  ❌ 취소됨")
        return None

    print(f"  이동 후: {pt2}")

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    print(f"  이동량: dx={dx}, dy={dy}")
    return dx, dy


def main():
    # config.json에서 포트/모니터 읽기
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        default_port = cfg["controller"]["port"]
        monitor_idx = cfg["capture"]["monitor"]
    except Exception:
        default_port = "COM4"
        monitor_idx = 1

    print("=" * 50)
    print("  피코 SCALE 측정 도구 (화면 클릭 방식)")
    print("=" * 50)
    port = input(f"COM 포트 [{default_port}]: ").strip() or default_port

    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(0.5)
        print(f"✅ {port} 연결됨")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        sys.exit(1)

    # PING
    send(ser, "PING")
    resp = ser.readline().decode("utf-8", "ignore").strip()
    print(f"PING → {resp}")

    print()
    print("측정 방법:")
    print("  - 캡처창에서 커서가 있는 위치를 직접 클릭")
    print("  - MOVE 전송 후 커서가 이동된 위치를 다시 클릭")
    print("  - 두 좌표 차이로 SCALE 자동 계산")
    input("\n준비되면 Enter → ")

    # 가로 측정 (MOVE:100:0)
    result_x = measure_axis(ser, "가로(X)", "MOVE:100:0", monitor_idx)

    # 세로 측정 (MOVE:0:100)
    result_y = measure_axis(ser, "세로(Y)", "MOVE:0:100", monitor_idx)

    ser.close()

    print()
    print("=" * 50)
    print("【결과】")

    scale_x = scale_y = None

    if result_x:
        px_x = abs(result_x[0])  # 가로 이동은 dx
        if px_x > 0:
            scale_x = round(100 / px_x, 4)
            print(f"  MOVE:100:0 → 실제 {px_x}px 이동")
            print(f"  SCALE_X = 100 / {px_x} = {scale_x}")

    if result_y:
        px_y = abs(result_y[1])  # 세로 이동은 dy
        if px_y > 0:
            scale_y = round(100 / px_y, 4)
            print(f"  MOVE:0:100 → 실제 {px_y}px 이동")
            print(f"  SCALE_Y = 100 / {px_y} = {scale_y}")

    if scale_x and scale_y:
        print()
        update = input("controller.py SCALE 자동 업데이트? (y/n): ").strip().lower()
        if update == "y":
            ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
            with open(ctrl_path, "r", encoding="utf-8") as f:
                code = f.read()
            code = re.sub(r"SCALE_X\s*=\s*[\d.]+", f"SCALE_X = {scale_x}", code)
            code = re.sub(r"SCALE_Y\s*=\s*[\d.]+", f"SCALE_Y = {scale_y}", code)
            with open(ctrl_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"✅ controller.py 업데이트 완료")
            print(f"   SCALE_X = {scale_x}")
            print(f"   SCALE_Y = {scale_y}")

    print("\n완료!")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
