"""
test_scale.py
-------------
피코 HID SCALE 측정 도구.

사용법:
  1. python test_scale.py
  2. 피코 COM 포트 입력
  3. 커서를 화면 아무데나 놓기
  4. Enter 누르면 MOVE:100:0 전송
  5. 커서가 실제로 몇 px 움직였는지 자로 재서 입력
  6. 정확한 SCALE 값 출력
"""

import sys
import time
import serial
import json
import os


def send(ser, text):
    ser.write((text + "\n").encode("utf-8"))
    ser.flush()
    time.sleep(0.1)


def main():
    # config.json에서 포트 읽기
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        default_port = cfg["controller"]["port"]
    except Exception:
        default_port = "COM4"

    print("=" * 50)
    print("  피코 SCALE 측정 도구")
    print("=" * 50)
    port = input(f"COM 포트 [{default_port}]: ").strip() or default_port

    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(0.5)
        print(f"✅ {port} 연결됨")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        sys.exit(1)

    # PING 확인
    send(ser, "PING")
    resp = ser.readline().decode("utf-8", "ignore").strip()
    print(f"PING → {resp}")

    print()
    print("─" * 50)
    print("【테스트 1】 가로 이동 (MOVE:100:0)")
    print("  커서를 화면 아무 곳에 놓고 Enter 누르세요")
    input("  준비되면 Enter → ")

    send(ser, "MOVE:100:0")
    print("  MOVE:100:0 전송됨")
    px_str = input("  실제로 몇 px 움직였나요? (자로 측정): ").strip()
    try:
        px_x = float(px_str)
        scale_x = 100 / px_x
        print(f"  → SCALE_X = 100 / {px_x} = {scale_x:.4f}")
    except Exception:
        scale_x = None
        print("  숫자를 입력해주세요")

    print()
    print("─" * 50)
    print("【테스트 2】 세로 이동 (MOVE:0:100)")
    print("  커서를 화면 아무 곳에 놓고 Enter 누르세요")
    input("  준비되면 Enter → ")

    send(ser, "MOVE:0:100")
    print("  MOVE:0:100 전송됨")
    px_str = input("  실제로 몇 px 움직였나요? (자로 측정): ").strip()
    try:
        px_y = float(px_str)
        scale_y = 100 / px_y
        print(f"  → SCALE_Y = 100 / {px_y} = {scale_y:.4f}")
    except Exception:
        scale_y = None
        print("  숫자를 입력해주세요")

    print()
    print("=" * 50)
    print("【결과】")
    if scale_x:
        print(f"  SCALE_X = {scale_x:.4f}")
    if scale_y:
        print(f"  SCALE_Y = {scale_y:.4f}")

    if scale_x and scale_y:
        avg = (scale_x + scale_y) / 2
        print(f"  평균    = {avg:.4f}")
        print()
        print("config.json controller.py 에 적용할 값:")
        print(f"  SCALE_X = {scale_x:.4f}")
        print(f"  SCALE_Y = {scale_y:.4f}")

        # config.json 자동 업데이트 여부
        update = input("\ncontroller.py SCALE 자동 업데이트? (y/n): ").strip().lower()
        if update == "y":
            ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
            with open(ctrl_path, "r", encoding="utf-8") as f:
                code = f.read()

            # SCALE_X, SCALE_Y 교체
            import re
            code = re.sub(r"SCALE_X\s*=\s*[\d.]+", f"SCALE_X = {scale_x:.4f}", code)
            code = re.sub(r"SCALE_Y\s*=\s*[\d.]+", f"SCALE_Y = {scale_y:.4f}", code)

            with open(ctrl_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"✅ controller.py SCALE 업데이트 완료")
            print(f"   SCALE_X = {scale_x:.4f}")
            print(f"   SCALE_Y = {scale_y:.4f}")

    ser.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
