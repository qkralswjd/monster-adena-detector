"""
main.py
-------
화면캡처 → YOLO → PICO 클릭 자동 공격봇.

동작:
  1. 화면 캡처 (mss 1920x1080)
  2. YOLO 탐지 (monster/adena)
  3. 타겟 추적 (TargetTracker - 한번 고정하면 사망 전까지 유지)
  4. 탐지된 프레임에서만 공격 (conf >= min_conf, cy >= min_cy, 쿨다운)
  5. 오버레이 갱신

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

    ccfg = cfg["controller"]
    acfg = cfg["attack"]
    tcfg = cfg["target"]

    cooldown = acfg["cooldown_sec"]
    drag_dx  = acfg["drag_dx"]
    drag_dy  = acfg["drag_dy"]
    hold_ms  = acfg["hold_ms"]
    min_conf = tcfg.get("min_conf", 0.35)
    min_cy   = tcfg.get("min_cy", 150)

    # ── 초기화 ────────────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"])
    print(f"[Init] 캡처: {cap.width}x{cap.height}  left={cap.left} top={cap.top}")

    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
    )

    if dummy_mode or not ccfg.get("enabled", True):
        ctrl = DummyController()
        ctrl.connect()
        print("[Init] DUMMY 모드")
    else:
        ctrl = PicoController(port=ccfg["port"], baudrate=ccfg["baudrate"])
        if not ctrl.connect():
            print("[Init] PICO 연결 실패 → DUMMY 모드")
            ctrl = DummyController()
            ctrl.connect()

    ov = OverlayWindow(
        mon_left = cap.left,
        mon_top  = cap.top,
        game_w   = cap.width,
        game_h   = cap.height,
        lb_x     = 0,
        ctrl     = ctrl,
    )
    ov.start()

    print(f"\n[Start] 자동 공격 시작. Ctrl+C 로 종료.")
    print(f"        쿨다운={cooldown}s  hold={hold_ms}ms  min_conf={min_conf}  min_cy={min_cy}\n")

    # ── 타겟 추적기 ───────────────────────────────────────────
    tracker = TargetTracker(
        miss_timeout = tcfg["miss_timeout_sec"],
        max_dist     = tcfg.get("max_dist", 350),
    )

    last_detections = []
    last_det_time   = 0.0
    last_attack_t   = 0.0
    prev_log_t      = 0.0
    LOG_INTERVAL    = 0.5

    try:
        while True:
            frame = cap.capture()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = det.detect(frame)
            now = time.time()

            monsters = [d for d in detections if d.class_id == 0]
            adenas   = [d for d in detections if d.class_id == 1]

            # ── 타겟 추적 ─────────────────────────────────────
            last_target, miss_elapsed = tracker.update(detections)

            # 오버레이용 탐지 결과 유지
            if detections:
                last_detections = detections
                last_det_time   = now
            elif now - last_det_time > tcfg["miss_timeout_sec"]:
                last_detections = []

            # ── 공격 ──────────────────────────────────────────
            # 조건: 타겟 있음 + 이번 프레임 실탐지 + conf/cy 통과 + 쿨다운
            if (last_target is not None
                    and miss_elapsed == 0.0              # 이번 프레임 실탐지
                    and last_target.confidence >= min_conf
                    and last_target.cy >= min_cy
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
                miss_elapsed = miss_elapsed,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
            )

            # ── 로그 (0.5초마다) ──────────────────────────────
            if now - prev_log_t >= LOG_INTERVAL:
                prev_log_t = now
                if monsters or adenas:
                    tgt_str = (f"cx={last_target.cx} cy={last_target.cy}"
                               if last_target else "없음")
                    print(f"── 몬스터 {len(monsters)}마리  아데나 {len(adenas)}  "
                          f"DET {det.fps:.1f}fps  타겟={tgt_str} ──")
                    for d in monsters:
                        mark = " ←" if (last_target and
                                        d.cx == last_target.cx and
                                        d.cy == last_target.cy) else ""
                        print(f"  [monster] cx={d.cx} cy={d.cy}  conf={d.confidence:.2f}{mark}")
                    for d in adenas:
                        print(f"  [adena]   cx={d.cx} cy={d.cy}  conf={d.confidence:.2f}")
                else:
                    if last_target and miss_elapsed > 0:
                        print(f"  타겟 miss {miss_elapsed:.1f}s  DET {det.fps:.1f}fps")
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
