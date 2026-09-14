import serial
import time
import ctypes
import ctypes.wintypes

s = serial.Serial('COM4', 115200, timeout=1)
time.sleep(0.5)

# 좌상단으로 리셋
s.write(b'MOVE:-9999:-9999\n')
s.flush()
time.sleep(0.5)
print('1. 좌상단 리셋 완료 - 커서가 좌상단에 있어야 함')

# 화면 중앙으로 이동
s.write(b'MOVE:960:540\n')
s.flush()
time.sleep(0.5)

pt = ctypes.wintypes.POINT()
ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
print(f'2. 커서 위치: x={pt.x}, y={pt.y}')
print(f'   중앙(960,540) 기준 오차: dx={pt.x-960}, dy={pt.y-540}')

# 특정 좌표 클릭 테스트 (화면 중앙 클릭)
print('3. 중앙 클릭 시도...')
s.write(b'CLICK:100\n')
s.flush()
time.sleep(0.3)
print(f'   응답: {s.readline()}')

s.close()
print('완료!')
