"""
overlay_main.py
---------------
YOLO 탐지 + 투명 오버레이 메인루프.

흐름:
  1. config.json 로드
  2. 게임 모니터 정보 + letterbox 오프셋 확인
  3. YOLO 탐지기 초기화
  4. 피코 컨트롤러 연결
  5. 투명 오버레이 창 시작 (별도 스레드)
  6. 메인루프: 캡처 → 탐지 → 추적 → 공격 → 오버레이 갱신

공격 로직:
  - 타겟 발견 → 첫 공격 즉시 실행
  - 타겟이 살아있는 동안 cooldown_sec 마다 재공격
  - 타겟 소실/사망 → 쿨다운 리셋 → 다음 타겟 선택

추적 로직 (find_matching):
  - IoU 우선 매칭 + IoU 없을 때 거리 기반 보조 추적
  - 빠르게 이동하는 몬스터도 max_dist 안이면 추적 유지

실행:
    python overlay_main.py

종료:
    Ctrl+C
"""

import sys
import os
import time
import json
import threading
import ctypes

sys.path.insert(0, os.path.dirname(__file__))

import mss
import cv2
import numpy as np

from screen_capture import ScreenCapture
from detector       import YOLODetector
from target_selector import select_nearest, find_matching
from controller     import PicoController, DummyController
from overlay_window import OverlayWindow


# ══════════════════════════════════════════════════════════════
#  letterbox 감지
# ══════════════════════════════════════════════════════════════

def get_letterbox_x(cfg, frame: np.ndarray) -> int:
    lb = cfg.get("letterbox", {})

    if lb.get("auto_detect", False):
        h, w = frame.shape[:2]
        mid_y = h // 2
        row = frame[mid_y]

        lb_left = 0
        for x in range(w):
            b, g, r = int(row[x][0]), int(row[x][1]), int(row[x][2])
            if r + g + b > 30:
                lb_left = x
                break

        lb_right = w
        for x in range(w - 1, -1, -1):
            b, g, r = int(row[x][0]), int(row[x][1]), int(row[x][2])
            if r + g + b > 30:
                lb_right = x
                break

        game_w = lb_right - lb_left
        print(f"[Letterbox] 자동감지: 왼쪽={lb_left}px  오른쪽={w-lb_right}px  게임너비={game_w}px")

        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        cfg["letterbox"]["x"]           = lb_left
        cfg["letterbox"]["game_width"]  = game_w
        cfg["letterbox"]["game_height"] = h
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)

        return lb_left

    val = lb.get("x", 0)
    print(f"[Letterbox] 설정값 사용: x={val}px")
    return val


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════

def main():
    # ── config 로드 ───────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # ── 캡처 초기화 ───────────────────────────────────────────
    mon_idx  = cfg["capture"]["monitor"]
    cap      = ScreenCapture(mon_idx)
    mon      = cap._monitor
    mon_left = mon["left"]
    mon_top  = mon["top"]
    cap_w    = mon["width"]
    cap_h    = mon["height"]
    print(f"[Capture] monitors[{mon_idx}]  left={mon_left} top={mon_top}  {cap_w}x{cap_h}")

    # ── 첫 프레임으로 letterbox 감지 ──────────────────────────
    print("[Init] 첫 프레임 캡처 중...")
    first_frame = None
    for _ in range(10):
        first_frame = cap.capture()
        if first_frame is not None:
            break
        time.sleep(0.1)

    if first_frame is None:
        print("[오류] 화면 캡처 실패. 종료합니다.")
        return

    lb_x   = get_letterbox_x(cfg, first_frame)
    game_w = cfg["letterbox"].get("game_width",  cap_w - lb_x * 2)
    game_h = cfg["letterbox"].get("game_height", cap_h)
    print(f"[Letterbox] lb_x={lb_x}  게임영역={game_w}x{game_h}")

    # ── YOLO 탐지기 ───────────────────────────────────────────
    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
    )

    # ── ROI ───────────────────────────────────────────────────
    rcfg = cfg["roi"]
    roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"]) \
          if rcfg.get("width", 0) > 0 else None
    print(f"[ROI] {roi if roi else '전체 화면'}")

    # ── 컨트롤러 ─────────────────────────────────────────────
    ccfg = cfg["controller"]
    if ccfg.get("enabled", True):
        ctrl = PicoController(
            port      = ccfg["port"],
            baudrate  = ccfg["baudrate"],
            mon_left  = mon_left,
            mon_top   = mon_top,
            lb_x      = lb_x,
            game_w    = game_w,
            game_h    = game_h,
        )
        if not ctrl.connect():
            print("[경고] 피코 연결 실패 → DummyController 사용")
            ctrl = DummyController()
            ctrl.connect()
    else:
        ctrl = DummyController()
        ctrl.connect()

    # ── 오버레이 창 시작 ──────────────────────────────────────
    overlay = OverlayWindow(
        mon_left = mon_left,
        mon_top  = mon_top,
        game_w   = game_w,
        game_h   = game_h,
        lb_x     = lb_x,
        ctrl     = ctrl,
    )
    overlay.start()
    print(f"[Overlay] 오버레이 시작: ({mon_left+lb_x},{mon_top})  {game_w}x{game_h}")
    time.sleep(0.5)  # tkinter 초기화 대기

    # ── 추적 파라미터 ─────────────────────────────────────────
    tcfg         = cfg["target"]
    MISS_TIMEOUT = tcfg.get("death_timeout_sec", 1.5)   # 소실 판정까지 대기 (초)
    IOU_THRESH   = tcfg.get("death_iou_thresh",  0.3)   # IoU 우선 채택 임계값
    MAX_DIST     = tcfg.get("track_max_dist",    120.0) # 거리 추적 최대 반경 (px)
    MIN_SCORE    = tcfg.get("track_min_score",   0.25)  # 혼합 스코어 최소값
    IOU_WEIGHT   = tcfg.get("track_iou_weight",  0.5)   # IoU 가중치
    DIST_WEIGHT  = tcfg.get("track_dist_weight", 0.5)   # 거리 가중치

    # ── 공격 설정 ─────────────────────────────────────────────
    acfg        = cfg["attack"]
    atk_enabled = acfg.get("enabled",      True)
    drag_dy     = acfg.get("drag_dy",      30)
    drag_dx     = acfg.get("drag_dx",      0)
    hold_ms     = acfg.get("hold_ms",      80)
    COOLDOWN    = acfg.get("cooldown_sec", 0.5)  # 재공격 쿨다운 (초)

    # ── 화면 중앙 (nearest 기준점) ────────────────────────────
    center_x = lb_x + game_w // 2   # 프레임 기준 중앙 X (letterbox 포함)
    center_y = game_h // 2

    # ── 피코 좌표 변환 ─────────────────────────────────────────
    def frame_to_game(cx: int, cy: int):
        """
        YOLO 프레임 좌표 → 게임 영역 내 픽셀 좌표.
          gx = cx - lb_x  (letterbox 제거)
          gy = cy          (상하 여백 없음)
        """
        return cx - lb_x, cy

    # ── 타겟 상태 변수 ────────────────────────────────────────
    current_target = None     # 현재 추적 중인 Detection
    miss_start     = None     # 소실 시작 시각
    prev_target_id = None     # 직전 타겟 식별용 (x,y 스냅샷)
    last_attack_t  = 0.0      # 마지막 공격 시각 (쿨다운 계산)

    print(f"[Main] 루프 시작. Ctrl+C 로 종료.")
    print(f"       오버레이 클릭으로 해당 위치 피코 클릭 가능.")
    print(f"       공격 쿨다운: {COOLDOWN}s  추적 반경: {MAX_DIST}px")
    print()

    try:
        while True:
            # ── 캡처 ──────────────────────────────────────────
            frame = cap.capture()
            if frame is None:
                continue

            # ── YOLO 탐지 ─────────────────────────────────────
            dets = det.detect(frame)

            # ROI 필터
            if roi:
                rx, ry, rw, rh = roi
                dets = [d for d in dets
                        if rx <= d.cx <= rx + rw and ry <= d.cy <= ry + rh]

            monsters = [d for d in dets if d.class_id == 0]

            # ── 타겟 추적 ─────────────────────────────────────
            if current_target is not None:
                matched = find_matching(
                    monsters, current_target,
                    iou_thresh  = IOU_THRESH,
                    max_dist    = MAX_DIST,
                    min_score   = MIN_SCORE,
                    iou_weight  = IOU_WEIGHT,
                    dist_weight = DIST_WEIGHT,
                )
                if matched:
                    current_target = matched
                    miss_start     = None
                else:
                    # 소실 타이머 시작
                    if miss_start is None:
                        miss_start = time.time()
                    elif time.time() - miss_start > MISS_TIMEOUT:
                        print(f"[Target] 타겟 소실/사망 → 해제")
                        current_target = None
                        miss_start     = None
                        prev_target_id = None
                        last_attack_t  = 0.0   # 쿨다운 리셋

            # 타겟 없으면 nearest 자동 선택
            if current_target is None and monsters:
                current_target = select_nearest(monsters, center_x, center_y)
                if current_target:
                    tgt_id = id(current_target)  # 새 객체 → 항상 다른 id
                    if tgt_id != prev_target_id:
                        prev_target_id = tgt_id
                        last_attack_t  = 0.0     # 새 타겟 → 쿨다운 리셋(즉시 공격)
                        gx0, gy0 = frame_to_game(current_target.cx, current_target.cy)
                        print(f"[Target] 새 타겟: 게임내({gx0},{gy0})")

            # ── 자동 공격 (쿨다운마다 재공격) ─────────────────
            now = time.time()
            if (atk_enabled
                    and current_target is not None
                    and not ctrl.is_attacking
                    and (now - last_attack_t) >= COOLDOWN):

                gx, gy = frame_to_game(current_target.cx, current_target.cy)
                ctrl.drag_attack(gx, gy,
                                 drag_dx=drag_dx,
                                 drag_dy=drag_dy,
                                 hold_ms=hold_ms)
                overlay.notify_attack(mon_left + lb_x + gx, mon_top + gy)
                last_attack_t = now

                is_first = (now - last_attack_t) < 0.01  # 첫 공격 여부 로그
                print(f"[Attack] 게임내({gx},{gy})  쿨다운={COOLDOWN}s")

            # ── 오버레이 갱신 ─────────────────────────────────
            miss_elapsed = (time.time() - miss_start
                            if miss_start else 0.0)
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
        print("\n[Main] Ctrl+C → 종료")
    finally:
        overlay.stop()
        if hasattr(ctrl, "disconnect"):
            ctrl.disconnect()
        print("[Main] 종료 완료")


if __name__ == "__main__":
    main()
