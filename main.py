"""
main.py
-------
[최종 v2] 화면캡처 → YOLO → PICO 클릭 자동 공격봇.

동작:
  1. 화면 캡처 (mss 1920x1080)
  2. YOLO 탐지 (monster/adena)
  3. 타겟 추적 (TargetTracker - 한번 고정하면 사망 전까지 유지)
  4. 공격 조건:
     - [실탐지 중] 타겟 탐지 됨 + conf>=min_conf + cy>=min_cy → 즉시 공격
     - [miss 중]   miss 시간이 miss_attack_timeout 이내 → 마지막 위치로 공격 계속
     - [사망 추정] miss_timeout 초과 → TargetTracker가 타겟 해제 + 새 타겟 선택
  5. 쿨다운 대기 (0.8초)
  6. 오버레이 갱신

좌표 원칙:
  cx, cy = YOLO 출력 그대로 → PICO 전달 (변환 없음)

실행:
  python main.py            # PICO 연결 (COM4)
  python main.py --dummy    # PICO 없이 로그만 (테스트)

종료:
  Ctrl+C
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector
from target_selector import TargetTracker
from controller import PicoController, DummyController
from overlay_window import OverlayWindow


def main():
    # ── 인자 파싱 ─────────────────────────────────────────────
    dummy_mode = "--dummy" in sys.argv

    # ── config 로드 ───────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    ccfg  = cfg["controller"]
    acfg  = cfg["attack"]
    tcfg  = cfg["target"]

    cooldown         = acfg["cooldown_sec"]          # 공격 쿨다운 (초)
    drag_dx          = acfg["drag_dx"]               # 드래그 X
    drag_dy          = acfg["drag_dy"]               # 드래그 Y (아래로)
    hold_ms          = acfg["hold_ms"]               # PRESS 유지 (ms)
    miss_timeout     = tcfg["miss_timeout_sec"]      # 타겟 소실 판단 시간 (초)
    min_conf         = tcfg.get("min_conf", 0.35)    # 공격 최소 confidence
    min_cy           = tcfg.get("min_cy", 150)       # UI 영역 제외 (cy 최솟값)

    # miss 중에도 마지막 위치로 공격할 최대 허용 시간
    # miss_timeout 보다 짧게 설정 → miss 초반에는 공격 유지
    miss_attack_timeout = miss_timeout * 0.6  # miss_timeout의 60% (예: 4초 → 2.4초)

    # ── 캡처 초기화 ───────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"])
    print(f"[Init] 캡처: {cap.width}x{cap.height}  left={cap.left} top={cap.top}")

    # ── YOLO 초기화 ───────────────────────────────────────────
    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
    )

    # ── PICO 컨트롤러 초기화 ──────────────────────────────────
    if dummy_mode or not ccfg.get("enabled", True):
        ctrl = DummyController()
        ctrl.connect()
        print("[Init] DUMMY 모드 (실제 클릭 없음)")
    else:
        ctrl = PicoController(port=ccfg["port"], baudrate=ccfg["baudrate"])
        if not ctrl.connect():
            print("[Init] PICO 연결 실패 → DUMMY 모드로 전환")
            ctrl = DummyController()
            ctrl.connect()

    # ── 오버레이 초기화 ───────────────────────────────────────
    ov = OverlayWindow(
        mon_left = cap.left,
        mon_top  = cap.top,
        game_w   = cap.width,
        game_h   = cap.height,
        lb_x     = 0,
        ctrl     = ctrl,
    )
    ov.start()

    print(f"[Init] 오버레이 시작")
    print()
    print("[Start] 자동 공격 시작. Ctrl+C 로 종료.")
    print(f"        쿨다운={cooldown}s  drag_dy={drag_dy}px  hold={hold_ms}ms")
    print(f"        miss_timeout={miss_timeout}s  miss_attack={miss_attack_timeout:.1f}s")
    print()

    # ── 타겟 추적기 ───────────────────────────────────────────
    tracker = TargetTracker(
        miss_timeout = miss_timeout,
        iou_thresh   = tcfg.get("iou_thresh", 0.2),
        max_dist     = tcfg.get("max_dist", 350),
    )

    # ── miss_timeout 버퍼 (오버레이용) ───────────────────────
    last_detections = []
    last_det_time   = 0.0

    # ── 공격 타이머 ───────────────────────────────────────────
    last_attack_t = 0.0

    # ── 로그 타이머 ───────────────────────────────────────────
    prev_log_t   = 0.0
    LOG_INTERVAL = 0.5

    try:
        while True:
            # ── 캡처 ──────────────────────────────────────────
            frame = cap.capture()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── YOLO 탐지 ─────────────────────────────────────
            detections = det.detect(frame)
            now = time.time()

            monsters = [d for d in detections if d.class_id == 0]
            adenas   = [d for d in detections if d.class_id == 1]

            # ── 타겟 추적 ─────────────────────────────────────
            last_target  = tracker.update(detections)
            miss_elapsed = tracker.miss_elapsed  # 0.0이면 탐지됨, >0이면 miss 중

            if detections:
                last_detections = detections
                last_det_time   = now
            else:
                if now - last_det_time > miss_timeout:
                    last_detections = []

            # ── PICO 공격 ─────────────────────────────────────
            #
            # 공격 가능 조건:
            #   A. 타겟이 존재함
            #   B. 쿨다운 지남
            #   C. 공격 중 아님
            #   D-1. [실탐지] miss_elapsed < 0.05 (이번 프레임 탐지) + conf/cy 조건
            #   D-2. [miss공격] miss_elapsed <= miss_attack_timeout → 마지막 위치로 공격
            #        (conf/cy 조건 완화 - 이미 고정된 타겟이므로 conf 재검사 불필요)
            #
            can_attack_base = (
                last_target is not None
                and now - last_attack_t >= cooldown
                and not ctrl.is_attacking
            )

            if can_attack_base:
                cx, cy = last_target.cx, last_target.cy

                # D-1: 실탐지 중 공격
                if (miss_elapsed < 0.05               # 이번 프레임 탐지됨
                        and detections                 # 실제 탐지 데이터 있음
                        and last_target.confidence >= min_conf
                        and cy >= min_cy):
                    print(f"[Attack] 실탐지 → ({cx},{cy})  "
                          f"conf={last_target.confidence:.2f}")
                    ctrl.drag_attack(cx, cy,
                                     drag_dx=drag_dx,
                                     drag_dy=drag_dy,
                                     hold_ms=hold_ms)
                    last_attack_t = now

                # D-2: miss 중 공격 (타겟 고정 유지하며 마지막 위치 공격)
                elif (0.05 <= miss_elapsed <= miss_attack_timeout
                        and cy >= min_cy):
                    print(f"[Attack] miss공격({miss_elapsed:.1f}s) → ({cx},{cy})")
                    ctrl.drag_attack(cx, cy,
                                     drag_dx=drag_dx,
                                     drag_dy=drag_dy,
                                     hold_ms=hold_ms)
                    last_attack_t = now

            # ── 오버레이 갱신 ─────────────────────────────────
            ov.update(
                detections   = last_detections,
                target       = last_target,
                miss_elapsed = miss_elapsed if not detections else 0.0,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
            )

            # ── 로그 (0.5초마다) ──────────────────────────────
            if now - prev_log_t >= LOG_INTERVAL:
                prev_log_t = now
                if monsters or adenas:
                    tgt_str = (f"TARGET cx={last_target.cx} cy={last_target.cy}"
                               if last_target else "TARGET none")
                    lock_str = "🔒" if tracker.is_locked else "  "
                    print(f"── {lock_str} 탐지 {len(monsters)}마리  아데나 {len(adenas)}  "
                          f"DET {det.fps:.1f}fps  {tgt_str} ──")
                    for d in monsters:
                        mark = " ← 추적중" if (
                            last_target and
                            d.cx == last_target.cx and d.cy == last_target.cy
                        ) else ""
                        print(f"  [monster] cx={d.cx} cy={d.cy}  "
                              f"conf={d.confidence:.2f}{mark}")
                    for d in adenas:
                        print(f"  [adena]   cx={d.cx} cy={d.cy}  "
                              f"conf={d.confidence:.2f}")
                else:
                    if last_target is not None and miss_elapsed > 0:
                        mode = ("miss공격가능" if miss_elapsed <= miss_attack_timeout
                                else "공격중단")
                        print(f"  타겟 miss {miss_elapsed:.1f}s [{mode}]  "
                              f"DET {det.fps:.1f}fps")
                    else:
                        print(f"  탐지 없음  DET {det.fps:.1f}fps")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        ctrl.stop()
        ctrl.disconnect()
        ov.stop()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
