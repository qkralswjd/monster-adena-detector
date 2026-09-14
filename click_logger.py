"""
click_logger.py
---------------
마우스 클릭 위치를 게임 프레임 좌표로 출력.

게임 모니터: mss monitors[1] (left=-1920)
클릭하면 → 프레임 좌표(cx, cy) + 화면비율 + YOLO 탐지 좌표와 비교 가능

실행: python click_logger.py
종료: Ctrl+C 또는 ESC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pynput import mouse
import mss
import json
import time

# config 읽기
cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

mon_idx = cfg["capture"]["monitor"]  # 1

# mss로 게임 모니터 정보 가져오기
with mss.mss() as sct:
    mon = sct.monitors[mon_idx]

MON_LEFT   = mon["left"]
MON_TOP    = mon["top"]
MON_WIDTH  = mon["width"]
MON_HEIGHT = mon["height"]

print("=" * 55)
print(f"  클릭 좌표 로거 시작")
print(f"  게임 모니터: left={MON_LEFT}, top={MON_TOP}")
print(f"  캡처 크기: {MON_WIDTH} x {MON_HEIGHT}")
print(f"  게임 화면 위에서 클릭하면 프레임 좌표 출력")
print(f"  종료: Ctrl+C")
print("=" * 55)

click_count = 0

def on_click(abs_x, abs_y, button, pressed):
    global click_count
    if not pressed:
        return
    if button != mouse.Button.left:
        return

    # 절대 좌표 → 게임 프레임 좌표 변환
    frame_x = abs_x - MON_LEFT
    frame_y = abs_y - MON_TOP

    # 게임 화면 밖이면 무시
    if not (0 <= frame_x <= MON_WIDTH and 0 <= frame_y <= MON_HEIGHT):
        print(f"  [클릭] 게임 화면 밖: 절대({abs_x},{abs_y})")
        return

    click_count += 1
    ratio_x = frame_x / MON_WIDTH
    ratio_y = frame_y / MON_HEIGHT

    print(f"[클릭 #{click_count:03d}] "
          f"프레임({frame_x:4d},{frame_y:4d})  "
          f"화면비=({ratio_x:.3f},{ratio_y:.3f})  "
          f"절대=({abs_x},{abs_y})")

with mouse.Listener(on_click=on_click) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        print(f"\n종료. 총 {click_count}번 클릭 기록")
