"""
main.py
-------
화면캡처 → YOLO → PICO 클릭 자동 공격봇.

상태머신:
  HUNTING     → 몬스터 탐지 + 공격
              → miss_timeout 초과 시 (몬스터 사망 추정) → ADENA_CHECK
  ADENA_CHECK → 마지막 몬스터 위치 주변에서 아데나 탐지 + 클릭 줍기
              → adena_timeout 초 후 또는 아데나 없으면 → HUNTING

좌표 원칙:
  YOLO는 ROI 기준 좌표 출력 → roi_offset_x/y 더해 절대 화면 좌표로 변환 → PICO 전달
  abs_x = roi_x + det_cx  (pico_image_autoclicker/pc/vision.py 방식과 동일)

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
    adcfg = cfg.get("adena", {})

    cooldown        = acfg["cooldown_sec"]
    hold_ms         = acfg["hold_ms"]
    click_pulse_ms  = acfg.get("click_pulse_ms", 20)   # CLICK 원자 명령 펄스
    min_conf        = tcfg.get("min_conf", 0.35)
    min_cy          = tcfg.get("min_cy", 150)

    # 아데나 줍기 설정
    adena_enabled       = adcfg.get("enabled", True)
    adena_timeout       = adcfg.get("check_timeout_sec", 4.0)   # ADENA_CHECK 최대 대기
    adena_search_radius = adcfg.get("search_radius_px", 250)     # 몬스터 사망 위치 주변 탐색 반경
    adena_click_delay   = adcfg.get("click_delay_sec", 0.4)      # 아데나 클릭 간격
    adena_min_conf      = adcfg.get("min_conf", 0.20)            # 아데나 최소 confidence

    # 하트비트 설정
    ping_interval   = ccfg.get("ping_interval_sec", 3.0)
    ping_fail_limit = ccfg.get("ping_fail_limit", 3)

    # ── 초기화 ────────────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"], roi=cfg.get("roi"))
    print(f"[Init] 캡처: {cap.width}x{cap.height}  left={cap.left} top={cap.top}")

    # ROI 오프셋 (YOLO 좌표 → 절대 화면 좌표 변환용)
    # pico_image_autoclicker vision.py 방식:
    #   center_x = roi_x + det_x   (ROI 내부좌표 + ROI 시작점 = 절대좌표)
    _roi = cfg.get("roi", {})
    roi_offset_x = _roi.get("x", 0) if _roi.get("enabled") else 0
    roi_offset_y = _roi.get("y", 0) if _roi.get("enabled") else 0
    print(f"[Init] ROI 오프셋: +({roi_offset_x}, {roi_offset_y})")

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
        ctrl = PicoController(
            port           = ccfg["port"],
            baudrate       = ccfg["baudrate"],
            click_pulse_ms = click_pulse_ms,
        )
        if not ctrl.connect():
            print("[Init] PICO 연결 실패 → DUMMY 모드")
            ctrl = DummyController()
            ctrl.connect()

    # 오버레이는 항상 모니터 전체 기준 (0,0)에서 시작
    import mss as _mss
    with _mss.mss() as _sct:
        _mon = _sct.monitors[cfg["capture"]["monitor"]]
        _mon_left = _mon["left"]
        _mon_top  = _mon["top"]
        _mon_w    = _mon["width"]
        _mon_h    = _mon["height"]

    ov = OverlayWindow(
        mon_left = _mon_left,
        mon_top  = _mon_top,
        game_w   = cap.width,
        game_h   = cap.height,
        lb_x     = 0,
        ctrl     = ctrl,
        roi      = cfg.get("roi"),
        mon_w    = _mon_w,
        mon_h    = _mon_h,
    )
    ov.start()

    print(f"\n[Start] 자동 공격 시작. Ctrl+C 로 종료.")
    print(f"        쿨다운={cooldown}s  pulse={click_pulse_ms}ms  "
          f"min_conf={min_conf}  min_cy={min_cy}")
    print(f"        하트비트: {ping_interval}s 마다, {ping_fail_limit}회 실패 시 중단\n")

    # ── 타겟 추적기 ───────────────────────────────────────────
    tracker = TargetTracker(
        miss_timeout = tcfg["miss_timeout_sec"],
        max_dist     = tcfg.get("max_dist", 350),
    )

    last_detections = []
    last_det_time   = 0.0
    last_attack_t   = 0.0
    last_adena_click_t = 0.0
    prev_log_t      = 0.0
    LOG_INTERVAL    = 0.5

    # ── 상태머신 ──────────────────────────────────────────────
    # 상태: "HUNTING" | "ADENA_CHECK"
    state               = "HUNTING"
    dead_monster_cx     = None   # 몬스터 사망 추정 위치
    dead_monster_cy     = None
    adena_check_start   = 0.0   # ADENA_CHECK 진입 시각
    adena_clicked_set   = set() # 이미 클릭한 아데나 (cx,cy) 중복 방지

    # 하트비트 상태
    last_ping_t     = time.time()
    ping_fail_count = 0

    try:
        while True:
            now = time.time()

            # ── PING/PONG 하트비트 (pico_image_autoclicker 방식) ──
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

            # ── 캡처 + 탐지 ───────────────────────────────────
            frame, is_new = cap.capture()
            if frame is None:
                time.sleep(0.001)
                continue

            # 새 프레임일 때만 추론 (같은 프레임 중복 추론 방지)
            if not is_new:
                time.sleep(0.001)
                continue

            detections = det.detect(frame)

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

            # ══════════════════════════════════════════════════
            #  상태머신
            # ══════════════════════════════════════════════════

            if state == "HUNTING":
                # ── 공격 ──────────────────────────────────────
                # 조건: 타겟 있음 + 이번 프레임 실탐지 + conf/cy 통과 + 쿨다운
                if (last_target is not None
                        and miss_elapsed == 0.0
                        and last_target.confidence >= min_conf
                        and last_target.cy >= min_cy
                        and now - last_attack_t >= cooldown
                        and not ctrl.is_attacking):

                    # ROI 기준 좌표 → 절대 화면 좌표
                    cx = last_target.cx + roi_offset_x
                    cy = last_target.cy + roi_offset_y
                    print(f"[Attack] → ({cx},{cy})  conf={last_target.confidence:.2f}")
                    ctrl.drag_attack(cx, cy, hold_ms=hold_ms)
                    last_attack_t = now

                # ── 몬스터 소실 → ADENA_CHECK 전환 ───────────
                # tracker가 miss_timeout 초과로 타겟을 None으로 리셋한 직후를 포착:
                # last_target이 None이고, 직전까지 타겟이 있었던 위치를 기억
                if (adena_enabled
                        and last_target is None
                        and dead_monster_cx is not None):
                    # 몬스터 사망 추정 → 아데나 확인 모드 진입
                    state = "ADENA_CHECK"
                    adena_check_start = now
                    adena_clicked_set.clear()
                    print(f"[State] HUNTING → ADENA_CHECK  "
                          f"사망위치=({dead_monster_cx},{dead_monster_cy})")

                # 타겟이 있는 동안 마지막 위치 갱신 (miss 상태 제외: 실탐지만)
                if last_target is not None and miss_elapsed == 0.0:
                    dead_monster_cx = last_target.cx
                    dead_monster_cy = last_target.cy

            elif state == "ADENA_CHECK":
                # ── 아데나 탐지 + 줍기 ────────────────────────
                check_elapsed = now - adena_check_start

                # 탐색 범위 내 아데나 필터링
                nearby_adenas = []
                for d in adenas:
                    if d.confidence < adena_min_conf:
                        continue
                    if dead_monster_cx is not None:
                        import math as _math
                        dist = _math.hypot(d.cx - dead_monster_cx,
                                           d.cy - dead_monster_cy)
                        if dist > adena_search_radius:
                            continue
                    nearby_adenas.append(d)

                # 아직 클릭 안 한 아데나 중 confidence 최고값 선택
                unclicked = [d for d in nearby_adenas
                             if (round(d.cx), round(d.cy)) not in adena_clicked_set]

                if unclicked and now - last_adena_click_t >= adena_click_delay:
                    target_adena = max(unclicked, key=lambda d: d.confidence)
                    # ROI 기준 좌표 → 절대 화면 좌표
                    ax = target_adena.cx + roi_offset_x
                    ay = target_adena.cy + roi_offset_y
                    print(f"[Adena] 줍기 클릭 → ({ax},{ay})  conf={target_adena.confidence:.2f}  "
                          f"남은시간={adena_timeout - check_elapsed:.1f}s")
                    ctrl.click(ax, ay, pulse_ms=click_pulse_ms)
                    adena_clicked_set.add((round(target_adena.cx), round(target_adena.cy)))
                    last_adena_click_t = now

                # ADENA_CHECK 종료 조건
                timeout_over  = check_elapsed >= adena_timeout
                no_more_adena = (not nearby_adenas) and check_elapsed >= 0.5  # 0.5초 여유

                if timeout_over or no_more_adena:
                    reason = "타임아웃" if timeout_over else "아데나 없음"
                    print(f"[State] ADENA_CHECK → HUNTING  ({reason}  {check_elapsed:.1f}s)")
                    state = "HUNTING"
                    dead_monster_cx = None
                    dead_monster_cy = None
                    # 트래커 리셋 (새 몬스터 탐색)
                    tracker.reset()

            # ── 오버레이 갱신 ─────────────────────────────────
            ov.update(
                detections   = last_detections,
                target       = last_target,
                miss_elapsed = miss_elapsed,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
                state        = state,
            )

            # ── 로그 (0.5초마다) ──────────────────────────────
            if now - prev_log_t >= LOG_INTERVAL:
                prev_log_t = now
                state_str = f"[{state}]"
                if monsters or adenas:
                    tgt_str = (f"cx={last_target.cx} cy={last_target.cy}"
                               if last_target else "없음")
                    print(f"── {state_str} 몬스터 {len(monsters)}마리  "
                          f"아데나 {len(adenas)}  DET {det.fps:.1f}fps  타겟={tgt_str} ──")
                    for d in monsters:
                        mark = " ←" if (last_target and
                                        d.cx == last_target.cx and
                                        d.cy == last_target.cy) else ""
                        print(f"  [monster] cx={d.cx} cy={d.cy}  conf={d.confidence:.2f}{mark}")
                    for d in adenas:
                        print(f"  [adena]   cx={d.cx} cy={d.cy}  conf={d.confidence:.2f}")
                else:
                    if last_target and miss_elapsed > 0:
                        print(f"  {state_str} 타겟 miss {miss_elapsed:.1f}s  DET {det.fps:.1f}fps")
                    else:
                        print(f"  {state_str} 탐지 없음  DET {det.fps:.1f}fps")

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        cap.stop()
        ctrl.stop()
        ctrl.disconnect()
        ov.stop()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
