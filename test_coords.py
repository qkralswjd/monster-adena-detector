"""
test_coords.py
--------------
[단계 1] 마우스 절대좌표 정확성 검증.

목적:
  YOLO / PICO 없이 ctypes 로 마우스를 움직여서
  "화면 좌표 x=824 → 실제로 x=824에 커서가 있는가?" 를 확인.

사용법:
  python test_coords.py           # 고정 좌표 순서대로 이동
  python test_coords.py --click   # 각 위치에서 실제 클릭도 수행

출력 예시:
  [Move] → (0, 0)      | 실제 커서: (0, 0)      | 오차: (0, 0)
  [Move] → (960, 540)  | 실제 커서: (960, 540)  | 오차: (0, 0)
  [Move] → (824, 400)  | 실제 커서: (824, 400)  | 오차: (0, 0)
"""

import sys
import time
import ctypes
import ctypes.wintypes

# ── Win32 래퍼 ──────────────────────────────────────────────
user32 = ctypes.windll.user32

def mouse_move(x: int, y: int):
    """ctypes 로 마우스 절대 이동 (Windows 화면 좌표)."""
    user32.SetCursorPos(x, y)

def mouse_click(x: int, y: int):
    """절대 이동 후 왼쪽 클릭."""
    user32.SetCursorPos(x, y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)   # MOUSEEVENTF_LEFTDOWN
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)   # MOUSEEVENTF_LEFTUP

def get_cursor_pos() -> tuple:
    """현재 커서 위치 반환 (x, y)."""
    pt = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


# ── 테스트 좌표 목록 ────────────────────────────────────────
TEST_POINTS = [
    (0,    0),      # 좌상단
    (960,  540),    # 화면 중앙 (1920x1080 기준)
    (1919, 1079),   # 우하단 (범위 안)
    (824,  400),    # 임의 좌표
    (240,  400),    # letterbox 경계 (검증용)
    (1680, 400),    # 오른쪽 letterbox 경계
    (960,  0),      # 상단 중앙
    (960,  1079),   # 하단 중앙
]


def run_test(do_click: bool = False):
    print("=" * 55)
    print(" 마우스 절대좌표 정확성 검증")
    print(" 목표좌표 → SetCursorPos → GetCursorPos 비교")
    print("=" * 55)
    print(f"{'목표':>16}  {'실제':>16}  {'오차':>12}")
    print("-" * 55)

    all_ok = True
    for tx, ty in TEST_POINTS:
        # 이동
        mouse_move(tx, ty)
        time.sleep(0.1)   # 이동 안정화 대기

        # 실제 커서 읽기
        ax, ay = get_cursor_pos()
        dx = ax - tx
        dy = ay - ty
        ok = (dx == 0 and dy == 0)
        if not ok:
            all_ok = False

        mark = "✓" if ok else "✗"
        print(f"  {mark}  ({tx:4d},{ty:4d})  →  ({ax:4d},{ay:4d})  "
              f"오차({dx:+d},{dy:+d})")

        # 클릭 모드
        if do_click:
            mouse_click(tx, ty)
            time.sleep(0.2)

        time.sleep(0.3)

    print("-" * 55)
    if all_ok:
        print("결과: 전부 정확 ✓  좌표 변환 필요 없음")
    else:
        print("결과: 오차 발생 ✗  DPI 스케일 또는 좌표 문제 확인 필요")
    print()
    print("다음 단계:")
    print("  python test_coords.py --click   로 실제 클릭 테스트")
    print("  python test_pico.py             로 PICO 절대좌표 테스트")


if __name__ == "__main__":
    do_click = "--click" in sys.argv
    if do_click:
        print("[경고] 클릭 모드. 3초 후 시작합니다. 게임창을 포커스하세요.")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
    run_test(do_click=do_click)
