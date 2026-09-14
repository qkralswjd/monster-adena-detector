"""
test_click.py
-------------
HID 이동 정확도 테스트 (클릭 없음).

실행 후 마우스 커서 위치를 직접 확인해서 오차 측정.
SCALE 튜닝 시 사용.

측정 결과:
  SCALE=0.5 기준: MOVE:1 → 실제 2.53px 이동
  정확한 SCALE = 1/2.53 = 0.395
"""

import serial
import time
import ctypes
import ctypes.wintypes

# ── SCALE 설정 ───────────────────────────────
# controller.py 의 SCALE_X/Y 와 동일하게 맞출 것
SCALE = 0.395

PORT = "COM4"
BAUD = 115200

s = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(0.5)
print(f"[Test] COM4 연결 완료  SCALE={SCALE}")


def get_cursor():
    """Windows 커서 위치 (피코HID와 연동 안 되지만 참고용)."""
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def send(cmd: str):
    s.write((cmd + "\n").encode())
    s.flush()


def move_abs(target_x: int, target_y: int, from_x: int, from_y: int):
    """from → target 으로 MOVE (픽셀 기준, SCALE 적용)."""
    dx = target_x - from_x
    dy = target_y - from_y
    sdx = int(dx * SCALE)
    sdy = int(dy * SCALE)
    send(f"MOVE:{sdx}:{sdy}")
    time.sleep(0.6)
    return sdx, sdy


print()
print("=== HID 이동 정확도 테스트 ===")
print("각 단계 후 실제 커서 위치를 확인하세요")
print()

# ── 1. 좌상단 리셋 ───────────────────────────
print("1. 좌상단 리셋 (MOVE:-9999:-9999)...")
send("MOVE:-9999:-9999")
time.sleep(1.0)
x, y = get_cursor()
print(f"   Windows커서: ({x},{y}) → (0,0) 근처여야 함")
cur_x, cur_y = 0, 0

# ── 2. 중앙(960,540)으로 이동 ─────────────────
print()
print("2. 중앙(960,540)으로 이동...")
sdx, sdy = move_abs(960, 540, cur_x, cur_y)
print(f"   전송: MOVE:{sdx}:{sdy}")
x, y = get_cursor()
print(f"   Windows커서: ({x},{y}) → 목표(960,540)  오차:({x-960},{y-540})")
cur_x, cur_y = 960, 540

# ── 3. (300,200)으로 이동 ────────────────────
print()
print("3. (300,200)으로 이동...")
sdx, sdy = move_abs(300, 200, cur_x, cur_y)
print(f"   전송: MOVE:{sdx}:{sdy}")
x, y = get_cursor()
print(f"   Windows커서: ({x},{y}) → 목표(300,200)  오차:({x-300},{y-200})")
cur_x, cur_y = 300, 200

# ── 4. (1500,700)으로 이동 ───────────────────
print()
print("4. (1500,700)으로 이동...")
sdx, sdy = move_abs(1500, 700, cur_x, cur_y)
print(f"   전송: MOVE:{sdx}:{sdy}")
x, y = get_cursor()
print(f"   Windows커서: ({x},{y}) → 목표(1500,700)  오차:({x-1500},{y-700})")
cur_x, cur_y = 1500, 700

# ── 5. 중앙(960,540)으로 복귀 ────────────────
print()
print("5. 중앙(960,540)으로 복귀...")
sdx, sdy = move_abs(960, 540, cur_x, cur_y)
print(f"   전송: MOVE:{sdx}:{sdy}")
x, y = get_cursor()
print(f"   Windows커서: ({x},{y}) → 목표(960,540)  오차:({x-960},{y-540})")

print()
print("=== 완료! (클릭 없음) ===")
print(f"오차가 ±20px 이내면 SCALE={SCALE} OK")
print(f"오차가 크면 → SCALE을 조정하세요")
print(f"  커서가 목표보다 더 많이 이동 → SCALE 줄이기 (현재:{SCALE} → {SCALE*0.9:.3f})")
print(f"  커서가 목표에 못 미침      → SCALE 늘리기 (현재:{SCALE} → {SCALE*1.1:.3f})")

s.close()
