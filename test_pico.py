"""
test_pico.py
------------
PICO SCALE 측정 도구 (리셋 없는 버전).

GetCursorPos() 로 현재 커서 위치 읽고
목표 좌표까지 이동 후 실제 도착 위치로 SCALE 역산.

사용법:
  python test_pico.py           # SCALE 측정
  python test_pico.py --port COM3
"""

import sys
import time
import ctypes
import ctypes.wintypes

from controller import PicoController, DummyController


def get_cursor_pos() -> tuple:
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


def wait_cursor_stable(timeout=3.0) -> tuple:
    """커서가 멈출 때까지 대기 후 위치 반환."""
    prev = (-1, -1)
    stable = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.05)
        cur = get_cursor_pos()
        if cur == prev:
            stable += 1
            if stable >= 4:  # 0.2초 연속 안 움직임
                return cur
        else:
            stable = 0
        prev = cur
    return get_cursor_pos()


def measure_scale(ctrl):
    """
    현재 커서 위치 기준으로 3개 지점 이동 후 SCALE 역산.
    리셋 불필요. GetCursorPos 기반.
    """
    print("\n[SCALE 측정]")
    print("커서를 화면 중앙 근처에 놓고 기다리세요...\n")
    time.sleep(2.0)

    # 측정 지점 (이동 거리가 클수록 정확)
    targets = [
        (960,  540),
        (480,  270),
        (1440, 810),
    ]

    scale_x_list = []
    scale_y_list = []

    for i, (tx, ty) in enumerate(targets):
        # 현재 커서 위치 읽기
        cx, cy = get_cursor_pos()
        print(f"  [{i+1}/3] 현재({cx},{cy}) → 목표({tx},{ty})")

        # 목표까지 필요한 실제 픽셀 차이
        dx = tx - cx
        dy = ty - cy

        # 현재 SCALE로 HID 값 계산
        hid_x = round(dx * ctrl.SCALE_X)
        hid_y = round(dy * ctrl.SCALE_Y)

        # PICO로 전송
        ctrl._send(f"MOVE:{hid_x}:{hid_y}")

        # 이동 완료 대기
        steps = max(abs(hid_x), abs(hid_y)) / 127 + 1
        wait  = steps * 0.008 + 0.15
        time.sleep(wait)

        # 실제 도착 위치
        ax, ay = wait_cursor_stable()
        ex = ax - tx
        ey = ay - ty
        print(f"       실제도착({ax},{ay})  오차({ex:+d},{ey:+d})")

        # SCALE 역산
        # 실제로 이동된 픽셀 = ax - cx
        # 필요했던 HID = hid_x
        # 실제 SCALE = hid_x / (ax - cx)
        moved_x = ax - cx
        moved_y = ay - cy
        if abs(moved_x) > 10:
            real_sx = hid_x / moved_x
            scale_x_list.append(real_sx)
        if abs(moved_y) > 10:
            real_sy = hid_y / moved_y
            scale_y_list.append(real_sy)

        time.sleep(0.5)

    print()
    if not scale_x_list or not scale_y_list:
        print("측정 실패 (이동 거리가 너무 작음)")
        return

    new_sx = round(sum(scale_x_list) / len(scale_x_list), 4)
    new_sy = round(sum(scale_y_list) / len(scale_y_list), 4)

    print("=" * 50)
    print(f"  현재 SCALE_X = {ctrl.SCALE_X}  →  측정값 = {new_sx}")
    print(f"  현재 SCALE_Y = {ctrl.SCALE_Y}  →  측정값 = {new_sy}")
    print()
    print("  controller.py 수정:")
    print(f"    SCALE_X = {new_sx}")
    print(f"    SCALE_Y = {new_sy}")
    print("=" * 50)


if __name__ == "__main__":
    args  = sys.argv[1:]
    port  = "COM4"
    dummy = "--dummy" in args

    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = args[idx + 1]

    if dummy:
        ctrl = DummyController()
        ctrl.connect()
    else:
        ctrl = PicoController(port=port)
        if not ctrl.connect():
            print("PICO 연결 실패")
            sys.exit(1)

    print(f"포트: {port}")
    print("3초 후 시작합니다.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    measure_scale(ctrl)
    ctrl.disconnect()
