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

# recorder는 --record 플래그 없이는 완전 no-op (_NullRecorder)
from recorder import init_recorder, get_recorder

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
    if "--help" in sys.argv or "-h" in sys.argv:
        print("사용법: python main.py [옵션]")
        print("")
        print("옵션:")
        print("  (없음)         PICO 연결 후 ATTACKING_DUMMY 시작")
        print("  --dummy        PICO 없이 더미 컨트롤러 (로그 전용 테스트)")
        print("  --skip-dummy   ATTACKING_DUMMY 건너뛰고 MOVE_TO_HUNT_ZONE 시작")
        print("  --hunt-only    바로 HUNTING 상태에서 시작")
        print("  --record       화면 녹화 + 콘솔 로그 + 이벤트 JSON 동시 기록")
        print("                 test_runs/YYYY-MM-DD_HH-MM-SS/ 폴더에 저장")
        print("  --help, -h     이 도움말 출력")
        return

    dummy_mode  = "--dummy"       in sys.argv   # PICO 없이 더미 컨트롤러
    skip_dummy  = "--skip-dummy"  in sys.argv   # ATTACKING_DUMMY 건너뜀
    hunt_only   = "--hunt-only"   in sys.argv   # 바로 HUNTING
    record_mode = "--record"      in sys.argv   # 테스트 기록 (화면녹화+로그+이벤트)

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

    # ── 환경 요약 (진단용) ──────────────────────────────────────────
    # ※ mon_left/mon_top 은 현재 Pico 좌표 계산에 미포함(단일모니터=0이면 무관).
    #   단일 모니터: mon_left=0, mon_top=0 → 영향 없음
    #   멀티 모니터: mon_left≠0 → 좌표 오차 발생 가능 (이번 테스트에서는 미수정)
    try:
        import ctypes as _ctypes
        _dpi = _ctypes.windll.shcore.GetScaleFactorForDevice(0)  # % 단위 (100=100%)
        _dpi_str = f"{_dpi}%"
    except Exception:
        _dpi_str = "읽기실패(GetScaleFactorForDevice)"
    print(f"[Env] 모니터#{cfg['capture']['monitor']}  "
          f"mon_left={mon_left} mon_top={mon_top}  "
          f"mon_size={mon_w}x{mon_h}  DPI={_dpi_str}")
    print(f"[Env] ※ mon_left/mon_top은 Pico 절대좌표에 미포함 — 단일모니터(0,0)이면 무관")

    # ── StateMachine 초기화 ────────────────────────────────────────
    # LevelDetector가 ROI 좌표만 mss로 직접 캡처 → 전체화면 캡처 불필요
    sm = StateMachine(
        config       = cfg,
        ctrl         = ctrl,
        press_key_fn = _press_key,
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
    # min_conf: config.json → target.min_conf 와 동일한 값을 Tracker에 전달.
    # Tracker 타겟 선택(select_by_confidence)과 main.py 공격 조건이
    # 동일한 confidence 기준을 공유하게 됨.
    tracker = TargetTracker(
        miss_timeout = tcfg["miss_timeout_sec"],
        max_dist     = tcfg.get("max_dist", 350),
        min_conf     = min_conf,   # config.json → target.min_conf (현재 0.20)
    )
    print(f"[Init] Tracker min_conf={min_conf} (config target.min_conf)")

    # ── 테스트 기록 초기화 (--record 플래그, config 로드 이후) ─────
    # monitor 번호를 config에서 읽어야 하므로 config 로드 후에 위치.
    # --record 없으면 get_recorder()는 완전 no-op(_NullRecorder).
    if record_mode:
        _base_dir = os.path.join(os.path.dirname(__file__), "test_runs")
        _monitor  = cfg.get("capture", {}).get("monitor", 1)
        init_recorder(base_dir=_base_dir, monitor_index=_monitor, record_fps=10.0)
        print("[REC] --record 모드 활성화")

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
    if sm.patrol_mover and hasattr(sm.patrol_mover, 'waypoints'):
        print(f"  patrol_wps     : {len(sm.patrol_mover.waypoints)}개")
    elif sm.patrol_mover:
        print(f"  patrol_wps     : 랜덤 순찰 모드")

    # ── 하트비트 ────────────────────────────────────────────────────
    last_ping_t     = time.time()
    ping_fail_count = 0

    # 로깅용
    last_detections = []
    last_det_time   = 0.0
    last_attack_t   = 0.0
    prev_log_t      = 0.0
    LOG_INTERVAL    = 0.5

    # ── 전투 상태 ────────────────────────────────────────────────────
    # drag_attack() 호출 후 게임 내 자동사냥이 계속되는 구간을 추적.
    # ctrl.is_attacking 은 HID 스레드 상태(~수백ms)이며 게임 전투와 무관.
    # 전투 종료 판정: combat_active=True 상태에서 tracker.update()가
    # (None, 0.0)을 반환할 때 (miss_timeout 3.5s 초과 → 타겟 소실 확인).
    combat_active         = False
    _combat_target_info   = ""   # 로그용 타겟 식별 문자열

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

                # ── 전투 종료 감지 ───────────────────────────────
                # combat_active=True 상태에서 tracker.update()가 (None, 0.0)을
                # 반환하는 시점 = miss_timeout(3.5s) 초과 후 tracker._target=None.
                #
                # ※ 이 신호의 정확한 의미: "3.5초간 타겟을 탐지하지 못함"
                #   = TARGET_LOST (타겟 소실). 사망 확정이 아님.
                #   현재 코드에는 실제 사망을 확정할 수 있는 신호가 없음:
                #     - 몬스터 HP바 감지 없음 (level_detector는 플레이어 HP만)
                #     - 사망 애니메이션 클래스 없음 (YOLO는 monster/adena 2클래스)
                #     - 가림·화면 이탈과 실제 사망을 구분하는 별도 로직 없음
                #   miss 중(elapsed>0)은 가림/이탈 가능성 → 전투 유지.
                #   miss_timeout(3.5s) 초과 후 타겟 재선택 안 됨 = 화면에서 완전 소실.
                #   이것을 현재 코드에서 사용 가능한 가장 신뢰할 수 있는 신호로 사용.
                if combat_active and last_target is None and miss_elapsed == 0.0:
                    print(f"[COMBAT] TARGET_LOST target={_combat_target_info} "
                          f"(miss_timeout {tcfg['miss_timeout_sec']}s 초과, "
                          f"사망 추정 — 가림/이탈과 구분 불가)")
                    get_recorder().log_event("TARGET_LOST", {
                        "target": _combat_target_info,
                        "miss_timeout_sec": tcfg["miss_timeout_sec"],
                    })
                    combat_active = False

                # ── 공격 실행 (combat_active=False일 때만) ───────
                # 조건 10개:
                #   기존 5개: target 존재, miss_elapsed==0.0, confidence,
                #             cy, cooldown, not is_attacking
                #   추가 1개: not combat_active (게임 자동사냥 중 새 공격 금지)
                if (not combat_active
                        and last_target is not None
                        and miss_elapsed == 0.0
                        and last_target.confidence >= min_conf
                        and last_target.cy >= min_cy
                        and now - last_attack_t >= cooldown
                        and not ctrl.is_attacking):
                    cx = last_target.cx + roi_offset_x
                    cy = last_target.cy + roi_offset_y
                    _combat_target_info = (f"cx={last_target.cx} "
                                           f"cy={last_target.cy} "
                                           f"conf={last_target.confidence:.2f}")
                    # ── 좌표 변환 상세 로그 ──────────────────────────────
                    # ※ mon_left/mon_top은 Pico 좌표에 미포함(이번 테스트에서 변경 없음)
                    print(f"[ATTACK] ROI=({last_target.cx},{last_target.cy}) "
                          f"OFFSET=({roi_offset_x},{roi_offset_y}) "
                          f"MON=({mon_left},{mon_top})[미사용] "
                          f"ABS=({cx},{cy})")

                    # ── 게임 창 foreground 확인 ───────────────────────────
                    # HID 입력은 foreground 창에 전달됨.
                    # 게임 창이 뒤에 있으면 입력이 다른 창으로 가거나 무시될 수 있음.
                    # ※ 이 확인은 로그 전용. 입력 구조 변경 없음.
                    try:
                        import ctypes as _ctypes_fg
                        _hwnd = _ctypes_fg.windll.user32.GetForegroundWindow()
                        _buf  = _ctypes_fg.create_unicode_buffer(256)
                        _ctypes_fg.windll.user32.GetWindowTextW(_hwnd, _buf, 256)
                        _fg_title = _buf.value.strip() or "(제목없음)"
                        print(f"[WINDOW] foreground='{_fg_title}'  hwnd={_hwnd}")
                    except Exception as _fe:
                        print(f"[WINDOW] foreground 확인 실패: {_fe}")

                    # 이벤트 순서: TARGET_SELECTED(타겟 확정) → COMBAT_START(공격 명령)
                    print(f"[Attack] → ({cx},{cy})  conf={last_target.confidence:.2f}")
                    get_recorder().log_event("TARGET_SELECTED", {
                        "cx":     last_target.cx,
                        "cy":     last_target.cy,
                        "conf":   round(last_target.confidence, 3),
                        "abs_cx": cx,
                        "abs_cy": cy,
                    })
                    print(f"[COMBAT] START target={_combat_target_info} "
                          f"→ 종료는 TARGET_LOST({tcfg['miss_timeout_sec']}s) 대기")
                    get_recorder().log_event("COMBAT_START", {
                        "cx":   last_target.cx,
                        "cy":   last_target.cy,
                        "conf": round(last_target.confidence, 3),
                    })
                    ctrl.drag_attack(cx, cy, hold_ms=hold_ms)
                    last_attack_t  = now
                    combat_active  = True
                elif combat_active and last_target is not None and miss_elapsed == 0.0:
                    # 전투 중 새 공격 요청이 왔으나 차단
                    if now - prev_log_t >= LOG_INTERVAL:
                        print(f"[COMBAT] BLOCK_NEW_ATTACK target={_combat_target_info}")
                        get_recorder().log_event("COMBAT_BLOCK_NEW_ATTACK", {
                            "target": _combat_target_info,
                        })

                # 로그
                if now - prev_log_t >= LOG_INTERVAL:
                    prev_log_t = now
                    if monsters or [d for d in detections if d.class_id == 1]:
                        tgt_str = (f"cx={last_target.cx} cy={last_target.cy}"
                                   if last_target else "없음")
                        print(f"── [HUNTING] 몬={len(monsters)}  "
                              f"DET {det.fps:.1f}fps  타겟={tgt_str}  "
                              f"combat={'ON' if combat_active else 'OFF'} ──")
            else:
                # HUNTING 아닌 상태 전환 시:
                # combat_active=False 일 때만 tracker.reset() 허용.
                # 전투 중(combat_active=True)인데 상태 전환이 일어난 경우는
                # reset 보류 → HUNTING 복귀 시 자연스럽게 해제.
                if (sm.state not in (BotState.HUNTING,)
                        and not combat_active
                        and tracker.target is not None):
                    tracker.reset()

            # ── StateMachine tick ─────────────────────────────────
            sm.update(frame, detections, combat_active=combat_active)

            # ── 오버레이 갱신 ────────────────────────────────────
            # LevelDetector가 캐시 반환 (실제 캡처는 내부에서 타이머 기반으로)
            hp_pct    = sm.level_det.read_hp()
            level_now = sm.level_det.read_level()

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
        cap.stop()
        ctrl.stop()
        ctrl.disconnect()
        ov.stop()
        # 테스트 기록 종료 (--record 없으면 no-op)
        get_recorder().stop()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
