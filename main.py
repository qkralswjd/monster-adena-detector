"""
main.py
-------
리니지 자동 공격봇 — StateMachine 기반 전체 흐름.

상태머신 (state_machine.py):
  IDLE → ATTACKING_DUMMY → MOVE_TO_HUNT_ZONE → HUNTING → LOOTING → DONE

  ATTACKING_DUMMY   : 허수아비 드래그 반복 공격 (레벨 달성까지)
  MOVE_TO_HUNT_ZONE : hunt_waypoints 7개 경유 이동
  HUNTING           : YOLO 몬스터 탐지·공격 + patrol_waypoints 순찰
  LOOTING           : 아데나 HSV 탐지·줍기
  DONE              : 완료

공통 (모든 상태):
  HP < hp_threshold%  → F4 물약 자동 사용
  F6 아이템           → 5분마다 자동 사용
  F7 아이템           → 8분마다 자동 사용

좌표 원칙:
  YOLO는 ROI 기준 좌표 출력 → roi_offset_x/y 더해 절대 화면 좌표로 변환 → PICO 전달

실행:
  python main.py            # PICO 연결 (COM4)
  python main.py --dummy    # PICO 없이 로그만 (테스트)
  python main.py --skip-dummy   # ATTACKING_DUMMY 건너뛰고 MOVE_TO_HUNT_ZONE 시작
  python main.py --hunt-only    # 바로 HUNTING 시작

설정:
  오버레이 실행 후:
    F1 + 우클릭  → 허수아비 좌표 등록
    F2 + 우클릭  → 웨이포인트 등록 (최대 10개)
    F3 + 드래그  → 레벨 UI ROI 지정
    F4 + 드래그  → HP바 ROI 지정
    Enter        → config.json 저장

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
from state_machine import StateMachine, BotState

import mss as _mss
import numpy as np


# ── 아이템 키 입력 ──────────────────────────────────────────────────
def _press_key(key: str):
    """F1~F12 키 입력 (pyautogui)."""
    try:
        import pyautogui
        pyautogui.press(key.lower())
        print(f"[Item] 키 입력: {key}")
    except Exception as e:
        print(f"[Item] 키 입력 실패 ({key}): {e}")


def _load_config(cfg_path: str) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ── 인자 파싱 ──────────────────────────────────────────────────
    dummy_mode  = "--dummy"       in sys.argv   # PICO 없이 더미 컨트롤러
    skip_dummy  = "--skip-dummy"  in sys.argv   # ATTACKING_DUMMY 건너뜀
    hunt_only   = "--hunt-only"   in sys.argv   # 바로 HUNTING

    # ── config 로드 ────────────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    cfg = _load_config(cfg_path)

    def reload_config():
        nonlocal cfg
        cfg = _load_config(cfg_path)
        sm.reload(cfg)
        print("[Config] 재로드 완료")

    # ── 파라미터 추출 ───────────────────────────────────────────────
    ccfg  = cfg["controller"]
    acfg  = cfg["attack"]
    tcfg  = cfg["target"]
    ldcfg = cfg.get("level_detector", {})

    cooldown       = acfg["cooldown_sec"]
    hold_ms        = acfg["hold_ms"]
    click_pulse_ms = acfg.get("click_pulse_ms", 20)
    min_conf       = tcfg.get("min_conf", 0.15)
    min_cy         = tcfg.get("min_cy", 150)
    ping_interval  = ccfg.get("ping_interval_sec", 3.0)
    ping_fail_limit= ccfg.get("ping_fail_limit", 3)

    # ── 초기화 ─────────────────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"], roi=cfg.get("roi"))
    print(f"[Init] 캡처: {cap.width}x{cap.height}  left={cap.left} top={cap.top}")

    _roi = cfg.get("roi", {})
    roi_offset_x = _roi.get("x", 0) if _roi.get("enabled") else 0
    roi_offset_y = _roi.get("y", 0) if _roi.get("enabled") else 0
    print(f"[Init] ROI 오프셋: +({roi_offset_x}, {roi_offset_y})")

    dcfg2 = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg2["model"],
        confidence    = dcfg2["confidence"],
        iou_threshold = dcfg2["iou_threshold"],
        device        = dcfg2["device"],
        img_size      = dcfg2["img_size"],
    )

    if dummy_mode or not ccfg.get("enabled", True):
        ctrl = DummyController()
        ctrl.connect()
        print("[Init] DUMMY 모드")
    else:
        ctrl = PicoController(
            port           = ccfg["port"],
            baudrate       = ccfg["baudrate"],
            click_pulse_ms = click_pulse_ms,
        )
        if not ctrl.connect():
            print("[Init] PICO 연결 실패 → DUMMY 모드")
            ctrl = DummyController()
            ctrl.connect()

    # 모니터 정보
    with _mss.mss() as sct:
        mon      = sct.monitors[cfg["capture"]["monitor"]]
        mon_left = mon["left"]
        mon_top  = mon["top"]
        mon_w    = mon["width"]
        mon_h    = mon["height"]

    # 전체화면 캡처 (레벨/HP 인식용)
    _full_sct = _mss.mss()
    _full_mon = _full_sct.monitors[cfg["capture"]["monitor"]]

    def capture_full_frame():
        shot = _full_sct.grab(_full_mon)
        import cv2
        bgr = np.array(shot)[:, :, :3]
        bgr = bgr[:, :, ::-1].copy()  # BGRA → BGR
        return bgr

    # ── StateMachine 초기화 ────────────────────────────────────────
    sm = StateMachine(
        config             = cfg,
        ctrl               = ctrl,
        capture_full_frame = capture_full_frame,
        press_key_fn       = _press_key,
    )

    # ── 오버레이 ────────────────────────────────────────────────────
    dummy_pos  = cfg.get("dummy", {}).get("pos")
    waypoints  = cfg.get("movement", {}).get("waypoints", [])
    level_roi  = ldcfg.get("level_roi")
    hp_roi     = ldcfg.get("hp_roi")

    ov = OverlayWindow(
        mon_left        = mon_left,
        mon_top         = mon_top,
        game_w          = cap.width,
        game_h          = cap.height,
        lb_x            = 0,
        ctrl            = ctrl,
        roi             = cfg.get("roi"),
        mon_w           = mon_w,
        mon_h           = mon_h,
        config_path     = cfg_path,
        on_config_saved = reload_config,
    )
    ov.load_setup(
        dummy_pos  = dummy_pos,
        waypoints  = waypoints,
        level_roi  = level_roi,
        hp_roi     = hp_roi,
    )
    ov.start()

    # 타겟 추적기
    tracker = TargetTracker(
        miss_timeout = tcfg["miss_timeout_sec"],
        max_dist     = tcfg.get("max_dist", 350),
    )

    # ── StateMachine 시작 모드 선택 ────────────────────────────────
    if hunt_only:
        print("[Start] --hunt-only: 바로 HUNTING")
        sm.start_hunting()
    elif skip_dummy:
        print("[Start] --skip-dummy: MOVE_TO_HUNT_ZONE 시작")
        sm.start_at_hunt_zone()
    else:
        print("[Start] ATTACKING_DUMMY 시작 (레벨 달성까지)")
        sm.start()

    print(f"\n[Start] 자동 레벨링 시작. Ctrl+C 로 종료.")
    print(f"  허수아비 드래그: {sm.dummy_drag_from} → {sm.dummy_drag_to}")
    print(f"  hunt_waypoints : {len(sm.hunt_mover.waypoints) if sm.hunt_mover else 0}개")
    print(f"  patrol_wps     : {len(sm.patrol_mover.waypoints) if sm.patrol_mover else 0}개")

    # ── 하트비트 ────────────────────────────────────────────────────
    last_ping_t     = time.time()
    ping_fail_count = 0

    # 로깅용
    last_detections = []
    last_det_time   = 0.0
    last_attack_t   = 0.0
    prev_log_t      = 0.0
    LOG_INTERVAL    = 0.5

    try:
        while True:
            now = time.time()

            # ── 완료 체크 ─────────────────────────────────────────
            if sm.state == BotState.DONE:
                print("[Main] 상태머신 완료 → 종료")
                break

            # ── config 동적 반영 ──────────────────────────────────
            # (reload_config 콜백이 sm.reload()를 호출하므로 별도 처리 불필요)

            # ── PING 하트비트 ────────────────────────────────────
            if (isinstance(ctrl, PicoController)
                    and now - last_ping_t >= ping_interval):
                last_ping_t = now
                if ctrl.ping():
                    ping_fail_count = 0
                else:
                    ping_fail_count += 1
                    print(f"[Heartbeat] PING 실패 {ping_fail_count}/{ping_fail_limit}")
                    if ping_fail_count >= ping_fail_limit:
                        print("[Heartbeat] Pico 응답 없음 → 중단")
                        break

            # ── ROI 캡처 + YOLO 탐지 (HUNTING/LOOTING 상태) ──────
            detections = []
            frame      = None
            is_new     = False

            if sm.state in (BotState.HUNTING, BotState.LOOTING,
                            BotState.MOVE_TO_HUNT_ZONE):
                frame, is_new = cap.capture()
                if frame is not None and is_new:
                    detections = det.detect(frame)
                    if detections:
                        last_detections = detections
                        last_det_time   = now
                    elif now - last_det_time > tcfg["miss_timeout_sec"]:
                        last_detections = []

            # ── HUNTING: 몬스터 공격 로직 (main.py 담당) ─────────
            # StateMachine의 _update_hunting 은 순찰·아데나·레벨 체크를 담당.
            # 실제 공격(drag_attack)은 tracker + cooldown 기반으로 여기서 처리.
            if sm.state == BotState.HUNTING and frame is not None and is_new:
                monsters = [d for d in detections if d.class_id == 0]
                last_target, miss_elapsed = tracker.update(detections)

                if (last_target is not None
                        and miss_elapsed == 0.0
                        and last_target.confidence >= min_conf
                        and last_target.cy >= min_cy
                        and now - last_attack_t >= cooldown
                        and not ctrl.is_attacking):
                    cx = last_target.cx + roi_offset_x
                    cy = last_target.cy + roi_offset_y
                    print(f"[Attack] → ({cx},{cy})  conf={last_target.confidence:.2f}")
                    ctrl.drag_attack(cx, cy, hold_ms=hold_ms)
                    last_attack_t = now

                # 로그
                if now - prev_log_t >= LOG_INTERVAL:
                    prev_log_t = now
                    if monsters or [d for d in detections if d.class_id == 1]:
                        tgt_str = (f"cx={last_target.cx} cy={last_target.cy}"
                                   if last_target else "없음")
                        print(f"── [HUNTING] 몬={len(monsters)}  "
                              f"DET {det.fps:.1f}fps  타겟={tgt_str} ──")
            else:
                # HUNTING 아닌 상태 전환 시 1회만 리셋 (매 루프 호출 방지)
                if sm.state not in (BotState.HUNTING,) and tracker.target is not None:
                    tracker.reset()

            # ── StateMachine tick ─────────────────────────────────
            # SM.update() 내부에서 전체화면 캡처(grab) + HP/레벨 인식 수행
            # → full_frame을 SM에서 받아 오버레이에 재사용 (2중 캡처 제거)
            full_frame = sm.update(frame, detections)

            # ── 오버레이 갱신 ────────────────────────────────────
            if full_frame is not None:
                hp_pct    = sm.level_det.read_hp(full_frame)
                level_now = sm.level_det.read_level(full_frame)
            else:
                hp_pct    = None
                level_now = None

            ov.update(
                detections   = last_detections,
                target       = tracker.target,
                miss_elapsed = 0.0,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
                state        = sm.state_name,
                hp_pct       = hp_pct,
                level        = level_now,
            )

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        _full_sct.close()
        cap.stop()
        ctrl.stop()
        ctrl.disconnect()
        ov.stop()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
