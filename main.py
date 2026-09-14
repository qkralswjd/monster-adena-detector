"""
main.py
-------
[최종] 화면캡처 → YOLO → PICO 클릭 자동 공격봇.

동작:
  1. 화면 캡처 (mss 1920x1080)
  2. YOLO 탐지 (monster/adena)
  3. 타겟 선택 (confidence 최고)
  4. PICO drag_attack(cx, cy) 전송
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
from target_selector import select_by_confidence
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

    cooldown     = acfg["cooldown_sec"]       # 공격 쿨다운 (초)
    drag_dx      = acfg["drag_dx"]            # 드래그 X
    drag_dy      = acfg["drag_dy"]            # 드래그 Y (아래로)
    hold_ms      = acfg["hold_ms"]            # PRESS 유지 (ms)
    miss_timeout = tcfg["miss_timeout_sec"]   # 탐지 없을 때 박스 유지 (초)

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
    print()

    # ── miss_timeout 버퍼 ────────────────────────────────────
    last_detections = []
    last_target     = None
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

            if detections:
                last_detections = detections
                last_target     = select_by_confidence(detections)
                last_det_time   = now
                miss_elapsed    = 0.0
            else:
                miss_elapsed = now - last_det_time
                if miss_elapsed > miss_timeout:
                    last_detections = []
                    last_target     = None

            # ── PICO 공격 ─────────────────────────────────────
            # 조건: 타겟 있음 + 쿨다운 지남 + 공격 중 아님
            if (last_target is not None
                    and miss_elapsed == 0.0          # 현재 탐지된 타겟만
                    and now - last_attack_t >= cooldown
                    and not ctrl.is_attacking):

                cx, cy = last_target.cx, last_target.cy
                print(f"[Attack] → ({cx},{cy})  conf={last_target.confidence:.2f}")
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
                    print(f"── 탐지 {len(monsters)}마리  아데나 {len(adenas)}  "
                          f"DET {det.fps:.1f}fps ──")
                    for d in monsters:
                        mark = " ← TARGET" if (
                            last_target and
                            d.cx == last_target.cx and d.cy == last_target.cy
                        ) else ""
                        print(f"  [monster] cx={d.cx} cy={d.cy}  "
                              f"conf={d.confidence:.2f}{mark}")
                    for d in adenas:
                        print(f"  [adena]   cx={d.cx} cy={d.cy}  "
                              f"conf={d.confidence:.2f}")
                else:
                    if last_detections:
                        print(f"  탐지 없음 (유지 중 {now - last_det_time:.1f}s)  "
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
