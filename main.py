"""
main.py
-------
리니지 자동 공격봇 — 전체 상태머신.

상태머신:
  DUMMY_ATTACK  → 허수아비 고정 좌표 반복 공격 (레벨 5 달성까지)
              → 레벨 5 달성 → MOVING
  MOVING        → 웨이포인트 순서대로 클릭 이동
              → 마지막 웨이포인트 도달 → HUNTING
  HUNTING       → 몬스터 탐지 + 공격
              → miss_timeout 초과 → ADENA_CHECK
  ADENA_CHECK   → 아데나 탐지 + 줍기
              → timeout or 없음 → HUNTING

공통 (모든 상태):
  HP 50% 미만   → F4 (물약) 자동 사용
  F6 아이템     → 5분마다 자동 사용
  F7 아이템     → 8분마다 자동 사용

좌표 원칙:
  YOLO는 ROI 기준 좌표 출력 → roi_offset_x/y 더해 절대 화면 좌표로 변환 → PICO 전달
  abs_x = roi_x + det_cx  (pico_image_autoclicker/pc/vision.py 방식과 동일)

실행:
  python main.py            # PICO 연결 (COM4)
  python main.py --dummy    # PICO 없이 로그만 (테스트)

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
from level_detector import LevelDetector

import mss as _mss
import numpy as np


# ── 아이템 키 입력 (Windows SendInput) ──────────────────────
def _press_key(key: str):
    """
    F1~F12 키 입력.
    pyautogui 사용 (PICO 없이 키보드 직접 입력).
    """
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
    # ── 인자 파싱 ─────────────────────────────────────────────
    dummy_mode = "--dummy" in sys.argv

    # ── config 로드 ───────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    cfg = _load_config(cfg_path)

    def reload_config():
        nonlocal cfg
        cfg = _load_config(cfg_path)
        print("[Config] 재로드 완료")

    ccfg  = cfg["controller"]
    acfg  = cfg["attack"]
    tcfg  = cfg["target"]
    adcfg = cfg.get("adena", {})
    dcfg  = cfg.get("dummy", {})
    mcfg  = cfg.get("movement", {})
    ldcfg = cfg.get("level_detector", {})
    icfg  = cfg.get("items", {})

    # ── 기본 파라미터 ─────────────────────────────────────────
    cooldown        = acfg["cooldown_sec"]
    hold_ms         = acfg["hold_ms"]
    click_pulse_ms  = acfg.get("click_pulse_ms", 20)
    min_conf        = tcfg.get("min_conf", 0.35)
    min_cy          = tcfg.get("min_cy", 150)

    # 아데나
    adena_enabled       = adcfg.get("enabled", True)
    adena_timeout       = adcfg.get("check_timeout_sec", 4.0)
    adena_search_radius = adcfg.get("search_radius_px", 250)
    adena_click_delay   = adcfg.get("click_delay_sec", 0.4)
    adena_min_conf      = adcfg.get("min_conf", 0.20)

    # 허수아비
    dummy_pos         = dcfg.get("pos")          # [x, y] or None
    dummy_cooldown    = dcfg.get("cooldown_sec", 0.8)

    # 웨이포인트
    waypoints         = mcfg.get("waypoints", [])   # [[x,y], ...]
    move_interval     = mcfg.get("click_interval_sec", 1.5)

    # 레벨/HP
    target_level      = ldcfg.get("target_level", 5)
    hp_threshold      = ldcfg.get("hp_threshold", 50.0)  # 0~100% 범위 (level_detector.py read_hp 반환값 기준)
    level_roi         = ldcfg.get("level_roi")
    hp_roi            = ldcfg.get("hp_roi")

    # 아이템
    hp_potion_key     = icfg.get("hp_potion_key", "F4")
    speed1_key        = icfg.get("speed1_key", "F6")
    speed1_interval   = icfg.get("speed1_interval_sec", 300)
    speed2_key        = icfg.get("speed2_key", "F7")
    speed2_interval   = icfg.get("speed2_interval_sec", 480)

    # 하트비트
    ping_interval     = ccfg.get("ping_interval_sec", 3.0)
    ping_fail_limit   = ccfg.get("ping_fail_limit", 3)

    # ── 초기화 ────────────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"], roi=cfg.get("roi"))
    print(f"[Init] 캡처: {cap.width}x{cap.height}  left={cap.left} top={cap.top}")

    # ROI 오프셋 (YOLO 좌표 → 절대 화면 좌표)
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
        mon = sct.monitors[cfg["capture"]["monitor"]]
        mon_left = mon["left"]
        mon_top  = mon["top"]
        mon_w    = mon["width"]
        mon_h    = mon["height"]

    # 레벨/HP 감지기
    level_det = LevelDetector(
        level_roi    = level_roi,
        hp_roi       = hp_roi,
        target_level = target_level,
    )

    # 오버레이
    ov = OverlayWindow(
        mon_left       = mon_left,
        mon_top        = mon_top,
        game_w         = cap.width,
        game_h         = cap.height,
        lb_x           = 0,
        ctrl           = ctrl,
        roi            = cfg.get("roi"),
        mon_w          = mon_w,
        mon_h          = mon_h,
        config_path    = cfg_path,
        on_config_saved= reload_config,
    )

    # 저장된 설정 오버레이에 로드 (마커 표시용)
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

    print(f"\n[Start] 자동 공격 시작. Ctrl+C 로 종료.")

    # ── 전체 화면 캡처 (레벨/HP 인식용) ─────────────────────
    _full_sct = _mss.mss()
    _full_mon = _full_sct.monitors[cfg["capture"]["monitor"]]

    def capture_full_frame():
        """전체 모니터 캡처 → BGR numpy."""
        shot = _full_sct.grab(_full_mon)
        import cv2
        bgr = np.array(shot)[:, :, :3]  # BGRA→BGR
        bgr = bgr[:, :, ::-1].copy()    # RGB→BGR
        return bgr

    # ── 상태머신 변수 ─────────────────────────────────────────
    # 시작 상태: dummy_pos 있으면 DUMMY_ATTACK, 없으면 HUNTING
    if dummy_pos:
        state = "DUMMY_ATTACK"
        print(f"[State] 시작: DUMMY_ATTACK  허수아비=({dummy_pos[0]},{dummy_pos[1]})")
    elif waypoints:
        state = "MOVING"
        print(f"[State] 시작: MOVING  웨이포인트={len(waypoints)}개")
    else:
        state = "HUNTING"
        print("[State] 시작: HUNTING")

    # DUMMY_ATTACK 변수
    last_dummy_t     = 0.0

    # MOVING 변수
    wp_index         = 0
    last_move_t      = 0.0

    # HUNTING 변수
    last_detections  = []
    last_det_time    = 0.0
    last_attack_t    = 0.0
    prev_log_t       = 0.0
    LOG_INTERVAL     = 0.5

    # ADENA_CHECK 변수
    last_adena_click_t  = 0.0
    dead_monster_cx     = None
    dead_monster_cy     = None
    adena_check_start   = 0.0
    adena_clicked_set   = set()

    # ── 아이템 타이머 ─────────────────────────────────────────
    last_speed1_t    = time.time()   # 시작하자마자 한 번 사용
    last_speed2_t    = time.time()
    last_hp_potion_t = 0.0
    HP_POTION_COOLDOWN = 5.0         # 물약 연속 사용 방지 (5초)

    # ── 하트비트 ──────────────────────────────────────────────
    last_ping_t      = time.time()
    ping_fail_count  = 0

    try:
        while True:
            now = time.time()

            # ── config reload 반영 ────────────────────────────
            # on_config_saved 콜백이 cfg를 갱신하므로
            # 매 루프마다 최신값 참조
            dcfg  = cfg.get("dummy", {})
            mcfg  = cfg.get("movement", {})
            ldcfg = cfg.get("level_detector", {})

            dummy_pos      = dcfg.get("pos")
            waypoints      = mcfg.get("waypoints", [])
            level_roi      = ldcfg.get("level_roi")
            hp_roi         = ldcfg.get("hp_roi")

            level_det.update_rois(level_roi, hp_roi)

            # ── PING 하트비트 ─────────────────────────────────
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

            # ── 전체화면 캡처 (레벨/HP 인식) ─────────────────
            full_frame = capture_full_frame()
            hp_pct     = level_det.read_hp(full_frame)
            level_now  = level_det.read_level(full_frame)

            # ── 아이템 자동 사용 (공통) ───────────────────────
            # HP 물약
            if (hp_pct is not None
                    and hp_pct < hp_threshold
                    and now - last_hp_potion_t >= HP_POTION_COOLDOWN):
                print(f"[Item] HP {hp_pct:.1f}% < {hp_threshold:.0f}% → {hp_potion_key} 사용")
                _press_key(hp_potion_key)
                last_hp_potion_t = now

            # 이속 아이템 1 (5분마다)
            if now - last_speed1_t >= speed1_interval:
                print(f"[Item] {speed1_key} 이속아이템 사용 ({speed1_interval//60}분 경과)")
                _press_key(speed1_key)
                last_speed1_t = now

            # 이속 아이템 2 (8분마다)
            if now - last_speed2_t >= speed2_interval:
                print(f"[Item] {speed2_key} 이속아이템 사용 ({speed2_interval//60}분 경과)")
                _press_key(speed2_key)
                last_speed2_t = now

            # ══════════════════════════════════════════════════
            #  상태머신
            # ══════════════════════════════════════════════════

            if state == "DUMMY_ATTACK":
                # ── 허수아비 반복 공격 ────────────────────────
                if dummy_pos and now - last_dummy_t >= dummy_cooldown:
                    dx, dy = dummy_pos[0], dummy_pos[1]
                    ctrl.drag_attack(dx, dy, hold_ms=hold_ms)
                    last_dummy_t = now
                    print(f"[Dummy] 공격 → ({dx},{dy})")

                # 레벨 5 달성 확인
                if level_det.is_target_level_reached(full_frame):
                    print(f"[State] DUMMY_ATTACK → MOVING  레벨={level_now}")
                    state    = "MOVING"
                    wp_index = 0
                    last_move_t = 0.0

                # dummy_pos 미설정이면 HUNTING으로 넘어감
                elif not dummy_pos:
                    print("[State] DUMMY_ATTACK → HUNTING  (허수아비 좌표 미설정)")
                    state = "HUNTING"

            elif state == "MOVING":
                # ── 웨이포인트 순서대로 클릭 이동 ────────────
                if not waypoints:
                    print("[State] MOVING → HUNTING  (웨이포인트 없음)")
                    state = "HUNTING"
                elif wp_index >= len(waypoints):
                    print(f"[State] MOVING → HUNTING  (웨이포인트 {len(waypoints)}개 완료)")
                    state = "HUNTING"
                    tracker.reset()
                elif now - last_move_t >= move_interval:
                    wx, wy = waypoints[wp_index]
                    ctrl.click(wx, wy, pulse_ms=click_pulse_ms)
                    print(f"[Move] 웨이포인트 {wp_index+1}/{len(waypoints)} → ({wx},{wy})")
                    last_move_t = now
                    wp_index   += 1

            elif state == "HUNTING":
                # ── 캡처 + 탐지 ───────────────────────────────
                frame, is_new = cap.capture()
                if frame is None or not is_new:
                    time.sleep(0.001)
                    # 오버레이만 갱신
                    ov.update(
                        detections   = last_detections,
                        target       = None,
                        miss_elapsed = 0.0,
                        det_fps      = det.fps,
                        cap_fps      = cap.fps,
                        state        = state,
                        hp_pct       = hp_pct,
                        level        = level_now,
                    )
                    continue

                detections = det.detect(frame)
                monsters   = [d for d in detections if d.class_id == 0]
                adenas     = [d for d in detections if d.class_id == 1]

                last_target, miss_elapsed = tracker.update(detections)

                if detections:
                    last_detections = detections
                    last_det_time   = now
                elif now - last_det_time > tcfg["miss_timeout_sec"]:
                    last_detections = []

                # 공격
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

                # 몬스터 소실 → ADENA_CHECK
                if (adena_enabled
                        and last_target is None
                        and dead_monster_cx is not None):
                    state = "ADENA_CHECK"
                    adena_check_start = now
                    adena_clicked_set.clear()
                    print(f"[State] HUNTING → ADENA_CHECK  "
                          f"사망위치=({dead_monster_cx},{dead_monster_cy})")

                # 마지막 타겟 위치 갱신
                if last_target is not None and miss_elapsed == 0.0:
                    dead_monster_cx = last_target.cx
                    dead_monster_cy = last_target.cy

                # 로그
                if now - prev_log_t >= LOG_INTERVAL:
                    prev_log_t = now
                    if monsters or adenas:
                        tgt_str = (f"cx={last_target.cx} cy={last_target.cy}"
                                   if last_target else "없음")
                        print(f"── [HUNTING] 몬스터 {len(monsters)}  "
                              f"아데나 {len(adenas)}  DET {det.fps:.1f}fps  타겟={tgt_str} ──")

                ov.update(
                    detections   = last_detections,
                    target       = last_target,
                    miss_elapsed = miss_elapsed,
                    det_fps      = det.fps,
                    cap_fps      = cap.fps,
                    state        = state,
                    hp_pct       = hp_pct,
                    level        = level_now,
                )

            elif state == "ADENA_CHECK":
                # ── 아데나 줍기 ───────────────────────────────
                frame, is_new = cap.capture()
                if frame is not None and is_new:
                    detections = det.detect(frame)
                    adenas     = [d for d in detections if d.class_id == 1]
                else:
                    adenas = []

                check_elapsed = now - adena_check_start

                # 탐색 범위 내 아데나 필터링
                import math as _math
                nearby_adenas = []
                for d in adenas:
                    if d.confidence < adena_min_conf:
                        continue
                    if dead_monster_cx is not None:
                        dist = _math.hypot(d.cx - dead_monster_cx,
                                           d.cy - dead_monster_cy)
                        if dist > adena_search_radius:
                            continue
                    nearby_adenas.append(d)

                unclicked = [d for d in nearby_adenas
                             if (round(d.cx), round(d.cy)) not in adena_clicked_set]

                if unclicked and now - last_adena_click_t >= adena_click_delay:
                    target_adena = max(unclicked, key=lambda d: d.confidence)
                    # ROI 기준 좌표 → 절대 화면 좌표
                    ax = target_adena.cx + roi_offset_x
                    ay = target_adena.cy + roi_offset_y
                    print(f"[Adena] 줍기 클릭 → ({ax},{ay})  conf={target_adena.confidence:.2f}  "
                          f"남은={adena_timeout - check_elapsed:.1f}s")
                    ctrl.click(ax, ay, pulse_ms=click_pulse_ms)
                    adena_clicked_set.add((round(target_adena.cx), round(target_adena.cy)))
                    last_adena_click_t = now

                timeout_over  = check_elapsed >= adena_timeout
                no_more_adena = (not nearby_adenas) and check_elapsed >= 0.5

                if timeout_over or no_more_adena:
                    reason = "타임아웃" if timeout_over else "아데나 없음"
                    print(f"[State] ADENA_CHECK → HUNTING  ({reason}  {check_elapsed:.1f}s)")
                    state = "HUNTING"
                    dead_monster_cx = None
                    dead_monster_cy = None
                    tracker.reset()

                ov.update(
                    detections   = last_detections,
                    target       = None,
                    miss_elapsed = 0.0,
                    det_fps      = det.fps,
                    cap_fps      = cap.fps,
                    state        = state,
                    hp_pct       = hp_pct,
                    level        = level_now,
                )

            # ── 짧은 대기 ─────────────────────────────────────
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
