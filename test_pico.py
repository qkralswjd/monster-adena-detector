"""
test_pico.py
------------
[단계 2] PICO 절대좌표 정확성 검증.

목적:
  PICO 로 화면 특정 좌표로 이동했을 때
  실제 커서가 그 좌표에 있는지 확인.

  YOLO 없이 PICO 좌표만 검증.

사용법:
  python test_pico.py             # 고정 좌표 순서대로 이동
  python test_pico.py --click     # 각 위치에서 실제 클릭도 수행
  python test_pico.py --port COM3 # 포트 지정

출력 예시:
  [Pico] 이동: (0,0) → (960,540)  Δ(960,540)  HID(379,213)
  [Check] 목표(960,540) | 실제 커서(960,540) | 오차(0,0) ✓

오차가 발생하면:
  - SCALE 값 조정 필요
  - 오차 = 실제 - 목표
  - 예: 목표960 실제800 → SCALE 너무 작음 → SCALE 올려야 함
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


def run_test(ctrl, do_click: bool = False):
    print("=" * 60)
    print(" PICO 절대좌표 정확성 검증")
    print(" PICO 이동 후 GetCursorPos 로 실제 위치 비교")
    print("=" * 60)
    print(f"{'목표':>16}  {'실제':>16}  {'오차':>12}")
    print("-" * 60)

    all_ok = True
    for tx, ty in TEST_POINTS:
        # PICO 이동
        if do_click:
            ctrl.drag_attack(tx, ty, drag_dx=0, drag_dy=0, hold_ms=50)
            time.sleep(0.3)
        else:
            ctrl._move_to(tx, ty)
            time.sleep(0.2)

        # 실제 커서 확인
        ax, ay = get_cursor_pos()
        dx = ax - tx
        dy = ay - ty
        ok = abs(dx) <= 2 and abs(dy) <= 2   # ±2px 허용
        if not ok:
            all_ok = False

        mark = "✓" if ok else "✗"
        print(f"  {mark}  ({tx:4d},{ty:4d})  →  ({ax:4d},{ay:4d})  "
              f"오차({dx:+d},{dy:+d})")

        time.sleep(0.5)

    print("-" * 60)
    if all_ok:
        print("결과: 전부 정확 ✓")
        print("      PICO 좌표 = 화면 좌표  일치 확인됨")
    else:
        print("결과: 오차 발생 ✗")
        print("      controller.py 의 SCALE 값 조정 필요")
        print(f"      현재 SCALE = {ctrl.SCALE}")
        print("      실제 > 목표: SCALE 내려야 함")
        print("      실제 < 목표: SCALE 올려야 함")


if __name__ == "__main__":
    args = sys.argv[1:]
    do_click = "--click" in args

    port = "COM4"
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = args[idx + 1]

    dummy = "--dummy" in args

    print(f"포트: {port}  클릭모드: {do_click}  더미: {dummy}")

    if dummy:
        ctrl = DummyController()
        ctrl.connect()
    else:
        ctrl = PicoController(port=port)
        if not ctrl.connect():
            print("PICO 연결 실패 → DummyController 로 대체")
            ctrl = DummyController()
            ctrl.connect()

    print("3초 후 시작합니다. 게임창을 포커스하세요.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    run_test(ctrl, do_click=do_click)

    ctrl.disconnect()
