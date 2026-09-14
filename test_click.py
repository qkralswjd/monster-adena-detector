import serial
import time
import ctypes
import ctypes.wintypes

SCALE = 0.5  # HID 감도 보정값

s = serial.Serial('COM4', 115200, timeout=1)
time.sleep(0.5)

def get_cursor():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def move(dx, dy):
    sdx = int(dx * SCALE)
    sdy = int(dy * SCALE)
    s.write(f'MOVE:{sdx}:{sdy}\n'.encode())
    s.flush()
    time.sleep(0.5)

# 1. 좌상단 리셋
print('1. 좌상단 리셋...')
s.write(b'MOVE:-9999:-9999\n')
s.flush()
time.sleep(1.0)
x, y = get_cursor()
print(f'   커서: ({x},{y}) → (0,0) 근처여야 함')

# 2. 중앙(960,540)으로 이동
print('2. 중앙(960,540)으로 이동...')
move(960, 540)
x, y = get_cursor()
print(f'   커서: ({x},{y}) → (960,540) 근처여야 함  오차:({x-960},{y-540})')

# 3. (300,200)으로 이동
print('3. (300,200)으로 이동...')
move(300-960, 200-540)
x, y = get_cursor()
print(f'   커서: ({x},{y}) → (300,200) 근처여야 함  오차:({x-300},{y-200})')

# 4. (1500,700)으로 이동
print('4. (1500,700)으로 이동...')
move(1500-300, 700-200)
x, y = get_cursor()
print(f'   커서: ({x},{y}) → (1500,700) 근처여야 함  오차:({x-1500},{y-700})')

s.close()
print('완료! (클릭 없음)')
