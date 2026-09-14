"""
main.py
-------
Monster Tracker - 클릭드래그 공격 + 아데나 자동 줍기 버전

전체 흐름:
  ┌─────────────────────────────────────────────────────────┐
  │  1. 화면 캡처 (mss, 모니터 1)                           │
  │  2. YOLOv8s 탐지 (monster / adena 2클래스)             │
  │                                                         │
  │  [공격 단계]                                            │
  │  3. 타겟 자동 선택 (중앙에서 가장 가까운 monster)        │
  │  4. IoU 매칭으로 같은 몬스터 추적                       │
  │  5. PRESS → 드래그 → RELEASE (공격)                    │
  │  6. 사망/소실 확정(1.5초) 시                            │
  │     → 사망 위치 기억 (record_kill)                      │
  │     → 아데나 수집 단계로 전환                           │
  │                                                         │
  │  [아데나 수집 단계]                                     │
  │  7. 사망 위치 반경 내 adena(class_id==1) 탐색           │
  │  8. adena 있으면 CLICK 으로 줍기                        │
  │  9. 타임아웃 or 최대 줍기 완료 → 다음 몬스터 선택       │
  └─────────────────────────────────────────────────────────┘

핫키 (백그라운드 모드):
  F9  : 공격 ON/OFF 토글
  F10 : 종료
  Ctrl+C : 강제종료
"""

import sys
import os
import time
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection
from target_selector import select_nearest, find_matching
from death_detector import DeathDetector
from controller import PicoController, DummyController
from adena_collector import AdenaCollector


# ─────────────────────────────────────────────
def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
# ─────────────────────────────────────────────


class MonsterTrackerApp:

    WIN = "Monster Tracker"

    def __init__(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        self._cfg = load_config(cfg_path)

        # ── 화면 캡처 ───────────────────────────────
        self._capture = ScreenCapture(self._cfg["capture"]["monitor"])

        # ── YOLO 탐지기 ─────────────────────────────
        dcfg = self._cfg["detector"]
        self._detector = YOLODetector(
            model_path    = dcfg["model"],
            confidence    = dcfg["confidence"],
            iou_threshold = dcfg["iou_threshold"],
            device        = dcfg["device"],
            img_size      = dcfg["img_size"],
            classes       = dcfg["classes"],   # None = 전 클래스 (monster+adena)
        )

        # ── 사망 감지기 ─────────────────────────────
        tcfg = self._cfg["target"]
        self._death_detector = DeathDetector(
            timeout_sec = tcfg["death_timeout_sec"],
            iou_thresh  = tcfg["death_iou_thresh"],
        )

        # ── mss 실제 캡처 크기 확인 ─────────────────
        # ⚠ 핵심: mss 캡처 크기 == YOLO 탐지 좌표 범위 == 피코 이동 목표 범위
        # 이 세 값이 일치해야 탐지된 cx/cy 로 정확히 클릭 가능
        mon = self._capture._monitor
        mss_w = mon["width"]
        mss_h = mon["height"]
        self._mon_left = mon["left"]
        self._mon_top  = mon["top"]
        print(f"[Capture] 게임모니터: left={self._mon_left}, top={self._mon_top} "
              f"| mss 캡처 크기: {mss_w}x{mss_h}")

        # config.json game.width/height 와 mss 실제 크기 불일치 경고
        gcfg = self._cfg.get("game", {})
        cfg_w = gcfg.get("width", mss_w)
        cfg_h = gcfg.get("height", mss_h)
        if cfg_w != mss_w or cfg_h != mss_h:
            print(f"[⚠ 경고] config game 해상도({cfg_w}x{cfg_h}) != "
                  f"mss 캡처 크기({mss_w}x{mss_h})")
            print(f"[⚠ 경고] 좌표 불일치 발생! config.json game.width/height를 "
                  f"{mss_w}x{mss_h}로 수정하거나 DPI 스케일 확인 필요")
        else:
            print(f"[Capture] ✅ mss 캡처 크기 == 게임 해상도 ({mss_w}x{mss_h}) → 좌표 일치")

        # 피코 컨트롤러에 mss 실제 크기 전달 (SCREEN_W/H 기준으로 리셋/이동)
        screen_w = mss_w
        screen_h = mss_h

        # ── 컨트롤러 (피코 HID) ─────────────────────
        ccfg = self._cfg["controller"]
        if ccfg["enabled"] and ccfg["type"] == "pico":
            self._ctrl = PicoController(
                port      = ccfg["port"],
                baudrate  = ccfg["baudrate"],
                screen_w  = screen_w,   # ← mss 실제 캡처 너비
                screen_h  = screen_h,   # ← mss 실제 캡처 높이
            )
            if not self._ctrl.connect():
                print("[Controller] 피코 연결 실패 → Dummy 모드")
                self._ctrl = DummyController()
                self._ctrl.connect()
        else:
            self._ctrl = DummyController()
            self._ctrl.connect()

        # ── 공격 설정 ───────────────────────────────
        acfg = self._cfg["attack"]
        self._attack_enabled  = acfg["enabled"]
        self._attack_cooldown = acfg["cooldown_sec"]
        self._drag_dx         = acfg["drag_dx"]
        self._drag_dy         = acfg["drag_dy"]
        self._drag_hold_ms    = acfg["drag_hold_ms"]
        self._aim_offset_y    = acfg.get("aim_offset_y", -20)
        self._last_attack_time = 0.0

        # ── 아데나 수집기 ───────────────────────────
        encfg = self._cfg["adena"]
        self._adena = AdenaCollector(
            enabled             = encfg["enabled"],
            search_radius       = encfg["search_radius"],
            collect_timeout_sec = encfg["collect_timeout_sec"],
            click_hold_ms       = encfg["click_hold_ms"],
            max_picks           = encfg["max_picks"],
            mon_left            = self._mon_left,
            mon_top             = self._mon_top,
        )

        # ── 상태 변수 ───────────────────────────────
        self._target: Optional[Detection] = None
        self._target_locked = False
        self._running = False

        # ── FPS 제한 ────────────────────────────────
        self._cap_fps        = self._cfg["capture"]["fps"]
        self._frame_interval = 1.0 / self._cap_fps

        # ── ROI ─────────────────────────────────────
        rcfg = self._cfg["roi"]
        if rcfg["width"] > 0 and rcfg["height"] > 0:
            self._capture.set_roi(rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])

        # ── 로그 타이머 ─────────────────────────────
        self._last_detect_log = 0.0

    # ══════════════════════════════════════════════
    #  타겟 갱신
    # ══════════════════════════════════════════════
    def _update_target(self, monsters: list, all_detections: list) -> bool:
        """
        매 프레임 타겟 갱신.

        Returns:
            True  = 사망 확정 (아데나 수집 단계로 전환 필요)
            False = 아직 생존 중 or 타겟 없음
        """
        frame_h, frame_w = self._last_frame_size

        # ── 타겟 없음 → 자동 선택 ──────────────────
        if self._target is None:
            if monsters:
                self._target = select_nearest(monsters, frame_w // 2, frame_h // 2)
                if self._target:
                    self._target_locked = True
                    self._death_detector.reset()
                    print(f"[Target] 자동선택: ({self._target.cx},{self._target.cy}) "
                          f"conf={self._target.confidence:.2f}")
            return False

        # ── 타겟 있음 → IoU 매칭 ───────────────────
        matched = find_matching(monsters, self._target,
                                self._cfg["target"]["death_iou_thresh"])
        if matched:
            self._target = matched
            self._death_detector.reset()
            return False

        # ── 소실 중 → 사망 판정 ────────────────────
        dead = self._death_detector.update(monsters, self._target)
        if dead:
            print(f"[Target] 사망/소실 확정!")
            return True   # ← 사망 신호

        return False

    # ══════════════════════════════════════════════
    #  아데나 수집 후 다음 타겟 선택
    # ══════════════════════════════════════════════
    def _on_monster_dead(self):
        """
        사망 확정 시 호출.
        1. 아데나 수집기에 위치 알림
        2. 타겟 상태 초기화 (수집 완료 후 다음 타겟 선택)
        """
        if self._target is not None:
            self._adena.record_kill(self._target.cx, self._target.cy)

        # 타겟 초기화 (수집 중에는 다음 타겟 선택 보류)
        self._target = None
        self._target_locked = False

    # ══════════════════════════════════════════════
    #  공격
    # ══════════════════════════════════════════════
    def _try_attack(self):
        """타겟이 있고 쿨다운 지났으면 드래그 공격."""
        if not self._attack_enabled:
            return
        if self._target is None:
            return

        now = time.time()
        if now - self._last_attack_time < self._attack_cooldown:
            return

        # 피코 이동 목표 = 프레임 내 좌표 그대로
        # (피코 리셋 후 커서 기준점 = 게임 화면 좌상단(0,0))
        # mon_left/top 은 더하지 않음 — 피코는 게임 화면 내 상대좌표로 동작
        x = self._target.cx
        y = self._target.cy + self._aim_offset_y

        print(f"[Attack] 드래그공격: 프레임({self._target.cx},{self._target.cy}) "
              f"→ 피코목표({x},{y})")

        self._ctrl.drag_attack(
            x       = x,
            y       = y,
            drag_dx = self._drag_dx,
            drag_dy = self._drag_dy,
            hold_ms = self._drag_hold_ms,
        )
        self._last_attack_time = now

    # ══════════════════════════════════════════════
    #  핫키
    # ══════════════════════════════════════════════
    def _setup_hotkeys(self):
        try:
            import keyboard
            keyboard.add_hotkey("F9",  self._toggle_attack)
            keyboard.add_hotkey("F10", self._request_stop)
            print("[핫키] F9=공격ON/OFF  F10=종료")
        except ImportError:
            print("[핫키] keyboard 모듈 없음 → pip install keyboard")
        except Exception as e:
            print(f"[핫키] 등록 실패: {e}")

    def _toggle_attack(self):
        self._attack_enabled = not self._attack_enabled
        print(f"[핫키] 공격 {'ON' if self._attack_enabled else 'OFF'}")

    def _request_stop(self):
        self._running = False
        print("[핫키] F10 → 종료 요청")

    # ══════════════════════════════════════════════
    #  메인 루프
    # ══════════════════════════════════════════════
    def run(self):
        show_window = self._cfg.get("display", {}).get("show_window", False)
        self._running = True
        self._last_frame_size = (1080, 1920)

        if show_window:
            import cv2
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN, 960, 540)
            print("[MonsterTracker] 창 모드. Q/ESC=종료")
        else:
            self._setup_hotkeys()
            print("[MonsterTracker] 백그라운드 모드 시작!")
            print("[MonsterTracker] F9=공격ON/OFF  F10=종료  Ctrl+C=강제종료")

        last_time = time.time()

        try:
            while self._running:
                # ── FPS 제한 ────────────────────────────
                now = time.time()
                elapsed = now - last_time
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                last_time = time.time()

                # ── 캡처 ────────────────────────────────
                frame = self._capture.capture()
                if frame is None:
                    continue
                self._last_frame_size = (frame.shape[0], frame.shape[1])

                # ── 탐지 (전 클래스: monster + adena) ──
                all_detections = self._detector.detect(frame)
                monsters = [d for d in all_detections if d.class_id == 0]
                adenas   = [d for d in all_detections if d.class_id == 1]

                # ── 5초마다 현황 로그 ───────────────────
                if time.time() - self._last_detect_log > 5.0:
                    target_str = (f"있음({self._target.cx},{self._target.cy})"
                                  if self._target else "없음")
                    adena_str = (f"수집중({self._adena.elapsed:.1f}s)"
                                 if self._adena.is_collecting else f"{len(adenas)}개")
                    print(f"[Status] monster={len(monsters)} | "
                          f"adena={adena_str} | "
                          f"타겟={target_str} | "
                          f"공격={'ON' if self._attack_enabled else 'OFF'}")
                    self._last_detect_log = time.time()

                # ══════════════════════════════════════
                #  아데나 수집 단계 (수집 중일 때)
                # ══════════════════════════════════════
                if self._adena.is_collecting:
                    done = self._adena.check_and_collect(all_detections, self._ctrl)
                    if done:
                        # 수집 완료 or 타임아웃 → 다음 타겟 선택
                        if monsters:
                            frame_h, frame_w = self._last_frame_size
                            self._target = select_nearest(
                                monsters, frame_w // 2, frame_h // 2)
                            if self._target:
                                self._target_locked = True
                                self._death_detector.reset()
                                print(f"[Target] 다음 타겟: "
                                      f"({self._target.cx},{self._target.cy})")
                    # 수집 중에는 공격 단계 건너뜀
                    if show_window:
                        self._render_window(frame, all_detections)
                    continue

                # ══════════════════════════════════════
                #  공격 단계
                # ══════════════════════════════════════
                dead = self._update_target(monsters, all_detections)

                if dead:
                    # 사망 확정 → 아데나 수집 시작
                    self._on_monster_dead()
                else:
                    # 생존 중 → 공격
                    self._try_attack()

                # ── 창 모드 시각화 ───────────────────────
                if show_window:
                    self._render_window(frame, all_detections)

        except KeyboardInterrupt:
            print("\n[MonsterTracker] Ctrl+C → 종료")
        finally:
            if show_window:
                import cv2
                cv2.destroyAllWindows()
            self._ctrl.stop()
            self._ctrl.disconnect()
            print("[MonsterTracker] 종료 완료")

    # ══════════════════════════════════════════════
    #  창 모드 렌더링 (show_window=true 일 때만)
    # ══════════════════════════════════════════════
    def _render_window(self, frame, all_detections):
        try:
            import cv2
            import visualizer
            miss = self._death_detector.miss_elapsed
            out = visualizer.draw(
                frame        = frame,
                detections   = all_detections,
                target       = self._target,
                miss_elapsed = miss,
                detector_fps = self._detector.fps,
                capture_fps  = self._capture.fps,
            )
            # 아데나 수집 중이면 화면에 표시
            if self._adena.is_collecting:
                cv2.putText(out, f"ADENA COLLECT {self._adena.elapsed:.1f}s",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 215, 255), 2)
            cv2.imshow(self.WIN, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._running = False
            elif key == ord("c"):
                self._target = None
                self._target_locked = False
                self._death_detector.reset()
                print("[Target] 해제")
        except Exception as e:
            print(f"[Render] 오류: {e}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = MonsterTrackerApp()
    app.run()
