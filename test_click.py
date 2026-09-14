"""
test_click.py
-------------
피코 없이 ctypes로 실제 클릭 좌표 확인용.
YOLO 탐지 → 오버레이 표시 → ctypes 클릭

실행:
    python test_click.py
"""

import sys
import os
import time
import json
import ctypes

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector       import YOLODetector
from target_selector import select_nearest, find_matching
from overlay_window import OverlayWindow

import mss
import numpy as np
import cv2


def get_letterbox_x(cfg, frame):
    lb = cfg.get("letterbox", {})
    if lb.get("auto_detect", False):
        h, w = frame.shape[:2]
        row = frame[h // 2]
        lb_left = 0
        for x in range(w):
            if sum(int(row[x][i]) for i in range(3)) > 30:
                lb_left = x
                break
        return lb_left
    return lb.get("x", 0)


def ctypes_click(sc_x, sc_y, hold_ms=80, drag_dy=30):
    """ctypes로 실제 클릭 (게임에 안 먹힐 수 있음 - 좌표 확인용)"""
    # 커서 이동
    ctypes.windll.user32.SetCursorPos(sc_x, sc_y)
    time.sleep(0.05)
    # PRESS
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    time.sleep(hold_ms / 1000.0)
    # 드래그
    if drag_dy != 0:
        ctypes.windll.user32.SetCursorPos(sc_x, sc_y + drag_dy)
        time.sleep(0.03)
    # RELEASE
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    mon_idx = cfg["capture"]["monitor"]
    cap = ScreenCapture(mon_idx)
    mon = cap._monitor
    mon_left = mon["left"]
    mon_top  = mon["top"]
    cap_w    = mon["width"]
    cap_h    = mon["height"]
    print(f"[Capture] monitors[{mon_idx}]  left={mon_left} top={mon_top}  {cap_w}x{cap_h}")

    # 첫 프레임으로 letterbox 감지
    first_frame = None
    for _ in range(10):
        first_frame = cap.capture()
        if first_frame is not None:
            break
        time.sleep(0.1)

    lb_x   = get_letterbox_x(cfg, first_frame)
    game_w = cfg["letterbox"].get("game_width",  cap_w - lb_x * 2)
    game_h = cfg["letterbox"].get("game_height", cap_h)
    print(f"[Letterbox] lb_x={lb_x}  게임영역={game_w}x{game_h}")

    # 게임 중앙 (Windows 절대좌표)
    center_gx = game_w // 2   # 게임 영역 내 중앙 x
    center_gy = game_h // 2   # 게임 영역 내 중앙 y
    center_sc_x = mon_left + lb_x + center_gx  # Windows 절대좌표
    center_sc_y = mon_top  + center_gy
    print(f"[Center] 게임중앙=({center_gx},{center_gy})  Windows절대=({center_sc_x},{center_sc_y})")

    # YOLO
    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
    )

    # ROI
    rcfg = cfg["roi"]
    roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"]) \
          if rcfg.get("width", 0) > 0 else None

    # 오버레이 (클릭 없이 표시만)
    overlay = OverlayWindow(
        mon_left = mon_left,
        mon_top  = mon_top,
        game_w   = game_w,
        game_h   = game_h,
        lb_x     = lb_x,
        ctrl     = None,   # 피코 없음
    )
    overlay.start()
    time.sleep(0.5)

    # 공격 설정
    acfg         = cfg["attack"]
    drag_dy      = acfg.get("drag_dy", 30)
    drag_dx      = acfg.get("drag_dx", 0)
    hold_ms      = acfg.get("hold_ms", 80)
    atk_cooldown = acfg.get("cooldown_sec", 0.5)

    current_target    = None
    miss_start        = None
    prev_target_id    = None
    attacked          = False   # 타겟당 1번만 클릭
    MISS_TIMEOUT      = cfg["target"].get("death_timeout_sec", 1.5)
    MISS_IOU_THRESH   = cfg["target"].get("death_iou_thresh",  0.3)

    center_x = lb_x + game_w // 2
    center_y = game_h // 2

    print(f"\n[Test] ctypes 클릭 테스트 시작. Ctrl+C 종료")
    print(f"       오버레이에 노란 원 = ctypes가 클릭하는 위치\n")

    try:
        while True:
            frame = cap.capture()
            if frame is None:
                continue

            dets = det.detect(frame)
            if roi:
                rx, ry, rw, rh = roi
                dets = [d for d in dets if rx <= d.cx <= rx+rw and ry <= d.cy <= ry+rh]
            monsters = [d for d in dets if d.class_id == 0]

            # 타겟 추적
            if current_target is not None:
                matched = find_matching(monsters, current_target, MISS_IOU_THRESH)
                if matched:
                    current_target = matched
                    miss_start = None
                else:
                    if miss_start is None:
                        miss_start = time.time()
                    elif time.time() - miss_start > MISS_TIMEOUT:
                        print(f"[Target] 소실 → 해제")
                        current_target = None
                        miss_start     = None
                        prev_target_id = None
                        attacked       = False

            if current_target is None and monsters:
                current_target = select_nearest(monsters, center_x, center_y)
                if current_target:
                    tgt_id = (current_target.x, current_target.y)
                    if tgt_id != prev_target_id:
                        prev_target_id = tgt_id
                        attacked       = False
                        print(f"[Target] 새 타겟: 게임내({current_target.cx - lb_x},{current_target.cy})")

            # ctypes 클릭 - 타겟당 1번만
            if current_target is not None and not attacked:
                attacked = True

                # 현재 프레임의 최신 좌표로 클릭
                gx   = current_target.cx - lb_x
                gy   = current_target.cy
                sc_x = mon_left + lb_x + gx
                sc_y = mon_top  + gy

                print(f"[Click] 게임내({gx},{gy})  Windows절대({sc_x},{sc_y})")
                overlay.notify_attack(sc_x, sc_y)
                ctypes_click(sc_x, sc_y, hold_ms=hold_ms, drag_dy=drag_dy)

            # 오버레이 갱신
            miss_elapsed = (time.time() - miss_start if miss_start else 0.0)
            overlay.update(
                detections   = dets,
                target       = current_target,
                miss_elapsed = miss_elapsed,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
                roi          = roi,
            )

            time.sleep(1 / cfg["capture"]["fps"])

    except KeyboardInterrupt:
        print("\n종료")
    finally:
        overlay.stop()


if __name__ == "__main__":
    main()
