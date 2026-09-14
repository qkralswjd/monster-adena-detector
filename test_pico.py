"""
test_pico.py
------------
[단계 2] PICO 절대좌표 정확성 검증.

사용법:
  python test_pico.py                      # 기본 좌표 검증
  python test_pico.py --port COM3          # 포트 지정
  python test_pico.py --dummy              # PICO 없이 로그만
  python test_pico.py --scale              # SCALE 자동 측정 모드

SCALE 측정 모드:
  리셋(0,0) 후 특정 좌표로 이동
  실제 커서 위치 읽어서 SCALE_X / SCALE_Y 역산
  → config.json 에 적용할 값 출력
"""

import sys
import time
import ctypes
import ctypes.wintypes

from controller import PicoController, DummyController


# ── Win32 커서 읽기 ──────────────────────────────────────────
def get_cursor_pos() -> tuple:
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


# ── 테스트 좌표 ──────────────────────────────────────────────
TEST_POINTS = [
    (0,    0),
    (960,  540),
    (1919, 1079),
    (824,  400),
    (240,  400),
    (1680, 400),
]


# ── 리셋 확인 ────────────────────────────────────────────────
def check_reset(ctrl):
    """
    리셋 후 실제 커서가 (0,0) 에 있는지 확인.
    (0,0) 이 아니면 리셋 실패 → 이후 모든 좌표가 틀림.
    """
    print("\n[리셋 확인]")
    ax, ay = get_cursor_pos()
    print(f"  리셋 후 실제 커서: ({ax},{ay})")
    if ax == 0 and ay == 0:
        print("  리셋 성공 ✓")
        return True
    else:
        print(f"  리셋 실패 ✗  오차: ({ax},{ay})")
        print("  → controller.py _reset() 대기시간 더 늘려야 함")
        return False


# ── SCALE 측정 모드 ──────────────────────────────────────────
def measure_scale(ctrl):
    """
    리셋(0,0) 후 두 지점으로 이동해서 SCALE_X / SCALE_Y 역산.
    """
    print("\n[SCALE 측정 모드]")
    print("  리셋 후 (960,540) 으로 이동해서 실제 위치 측정")

    # 리셋 확인
    ax0, ay0 = get_cursor_pos()
    if ax0 != 0 or ay0 != 0:
        print(f"  리셋 안됨: ({ax0},{ay0}) → 측정 불가")
        return

    # (960, 540) 으로 이동
    TARGET_X, TARGET_Y = 960, 540
    dx = TARGET_X - ctrl._cur_x
    dy = TARGET_Y - ctrl._cur_y
    hid_x = round(dx * ctrl.SCALE_X)
    hid_y = round(dy * ctrl.SCALE_Y)
    ctrl._send(f"MOVE:{hid_x}:{hid_y}")
    ctrl._cur_x = TARGET_X
    ctrl._cur_y = TARGET_Y
    time.sleep(0.3)

    ax, ay = get_cursor_pos()
    print(f"  목표: ({TARGET_X},{TARGET_Y})")
    print(f"  실제: ({ax},{ay})")
    print(f"  HID 전송: ({hid_x},{hid_y})")

    if hid_x != 0 and ax != 0:
        real_scale_x = round(hid_x / ax, 4)
        print(f"\n  현재 SCALE_X = {ctrl.SCALE_X}")
        print(f"  측정 SCALE_X = {real_scale_x}  (= HID{hid_x} / 실제{ax}px)")

    if hid_y != 0 and ay != 0:
        real_scale_y = round(hid_y / ay, 4)
        print(f"  현재 SCALE_Y = {ctrl.SCALE_Y}")
        print(f"  측정 SCALE_Y = {real_scale_y}  (= HID{hid_y} / 실제{ay}px)")

    print(f"\n  → controller.py 에서 수정:")
    if hid_x != 0 and ax != 0:
        print(f"    SCALE_X = {round(hid_x / ax, 4)}")
    if hid_y != 0 and ay != 0:
        print(f"    SCALE_Y = {round(hid_y / ay, 4)}")


# ── 기본 좌표 검증 ────────────────────────────────────────────
def run_test(ctrl):
    print("\n" + "=" * 60)
    print(" PICO 절대좌표 정확성 검증")
    print("=" * 60)
    print(f"{'목표':>16}  {'실제':>16}  {'오차':>12}")
    print("-" * 60)

    all_ok = True
    for tx, ty in TEST_POINTS:
        ctrl._move_to(tx, ty)
        time.sleep(0.2)

        ax, ay = get_cursor_pos()
        dx = ax - tx
        dy = ay - ty
        ok = abs(dx) <= 3 and abs(dy) <= 3
        if not ok:
            all_ok = False

        mark = "✓" if ok else "✗"
        print(f"  {mark}  ({tx:4d},{ty:4d})  →  ({ax:4d},{ay:4d})  "
              f"오차({dx:+d},{dy:+d})")
        time.sleep(0.3)

    print("-" * 60)
    if all_ok:
        print("결과: 전부 정확 ✓  PICO 좌표 = 화면 좌표 일치")
    else:
        print("결과: 오차 발생 ✗")
        print("      python test_pico.py --scale  로 SCALE 측정하세요")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    port  = "COM4"
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = args[idx + 1]

    dummy     = "--dummy" in args
    do_scale  = "--scale" in args

    print(f"포트: {port}  더미: {dummy}  SCALE측정: {do_scale}")

    if dummy:
        ctrl = DummyController()
        ctrl.connect()
    else:
        ctrl = PicoController(port=port)
        if not ctrl.connect():
            print("PICO 연결 실패 → DummyController 로 대체")
            ctrl = DummyController()
            ctrl.connect()

    print("3초 후 시작합니다.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # 리셋 확인
    reset_ok = check_reset(ctrl)

    if do_scale:
        measure_scale(ctrl)
    else:
        if not reset_ok:
            print("\n[경고] 리셋 실패 상태로 계속 진행합니다.")
            print("       결과가 틀릴 수 있습니다.")
        run_test(ctrl)

    ctrl.disconnect()
