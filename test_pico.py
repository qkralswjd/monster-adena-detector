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
    커서가 멈출 때까지 최대 5초 대기.
    """
    print("\n[리셋 확인]")
    print("  커서 이동 완료 대기 중...")

    # 커서가 멈출 때까지 대기 (최대 5초)
    prev_x, prev_y = -1, -1
    stable_count = 0
    for _ in range(50):   # 0.1s * 50 = 5초
        time.sleep(0.1)
        ax, ay = get_cursor_pos()
        if ax == prev_x and ay == prev_y:
            stable_count += 1
            if stable_count >= 3:   # 0.3초 연속 안 움직이면 완료
                break
        else:
            stable_count = 0
        prev_x, prev_y = ax, ay

    ax, ay = get_cursor_pos()
    print(f"  리셋 후 실제 커서: ({ax},{ay})")
    if ax <= 2 and ay <= 2:
        print("  리셋 성공 ✓")
        return True
    else:
        print(f"  리셋 실패 ✗  커서가 ({ax},{ay}) 에 있음")
        print("  → controller.py _reset() 전송 횟수/대기시간 늘려야 함")
        return False


# ── SCALE 측정 모드 ──────────────────────────────────────────
def measure_scale(ctrl):
    """
    리셋(0,0) 후 3개 지점으로 이동해서 SCALE_X/Y 역산.
    긴 거리 측정일수록 정확함 → (1919,1079) 우하단 사용.
    """
    print("\n[SCALE 측정 모드]")

    # 리셋 확인
    ax0, ay0 = get_cursor_pos()
    if ax0 > 2 or ay0 > 2:
        print(f"  리셋 안됨: ({ax0},{ay0}) → 측정 불가")
        return

    results = []
    # 3개 지점 측정: 중앙, 우하단, 임의
    targets = [
        (960,  540),
        (1919, 1079),
        (480,  270),
    ]

    for TARGET_X, TARGET_Y in targets:
        # (0,0) 으로 다시 리셋
        ctrl._send("MOVE:-9999:-9999")
        ctrl._send("MOVE:-9999:-9999")
        time.sleep(2.5)
        ctrl._cur_x = 0
        ctrl._cur_y = 0

        # 목표로 이동 (SCALE 적용)
        hid_x = round(TARGET_X * ctrl.SCALE_X)
        hid_y = round(TARGET_Y * ctrl.SCALE_Y)
        ctrl._send(f"MOVE:{hid_x}:{hid_y}")
        time.sleep(0.4)

        ax, ay = get_cursor_pos()
        print(f"  목표({TARGET_X},{TARGET_Y})  실제({ax},{ay})  HID({hid_x},{hid_y})")

        if ax > 0 and ay > 0:
            results.append((TARGET_X, TARGET_Y, ax, ay, hid_x, hid_y))

    if not results:
        print("  측정 실패")
        return

    # 평균 SCALE 계산
    sx_list = [hid_x / ax for (_, _, ax, _, hid_x, _) in results if ax > 0]
    sy_list = [hid_y / ay for (_, _, _, ay, _, hid_y) in results if ay > 0]

    new_scale_x = round(sum(sx_list) / len(sx_list), 4)
    new_scale_y = round(sum(sy_list) / len(sy_list), 4)

    print(f"\n  현재 SCALE_X={ctrl.SCALE_X}  측정 SCALE_X={new_scale_x}")
    print(f"  현재 SCALE_Y={ctrl.SCALE_Y}  측정 SCALE_Y={new_scale_y}")
    print(f"\n  → controller.py 에서 수정:")
    print(f"    SCALE_X = {new_scale_x}")
    print(f"    SCALE_Y = {new_scale_y}")


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
