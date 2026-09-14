import serial
import time
import ctypes
import ctypes.wintypes

s = serial.Serial('COM4', 115200, timeout=1)
time.sleep(0.5)

# 좌상단으로 리셋
print('1. 좌상단으로 리셋 중...')
s.write(b'MOVE:-9999:-9999\n')
s.flush()
time.sleep(1.0)  # 충분히 대기

pt = ctypes.wintypes.POINT()
ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
print(f'   리셋 후 커서: x={pt.x}, y={pt.y}  (0,0 근처여야 함)')

# 화면 중앙으로 이동
print('2. 중앙(960,540)으로 이동 중...')
s.write(b'MOVE:960:540\n')
s.flush()
time.sleep(1.0)

ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
print(f'   이동 후 커서: x={pt.x}, y={pt.y}  (960,540 근처여야 함)')
print(f'   오차: dx={pt.x-960}, dy={pt.y-540}')

# 특정 좌표로 이동 (500, 300)
print('3. (500,300)으로 이동 중...')
s.write(b'MOVE:-460:-240\n')
s.flush()
time.sleep(1.0)

ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
print(f'   이동 후 커서: x={pt.x}, y={pt.y}  (500,300 근처여야 함)')
print(f'   오차: dx={pt.x-500}, dy={pt.y-300}')

s.close()
print('완료! (클릭 없음)')
