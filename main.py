"""
main.py
-------
Monster Tracker - 클릭드래그 공격 버전

흐름:
  1. 화면 캡처 (mss, 모니터 1)
  2. YOLOv8s 탐지 (monster 클래스만, class_id==0)
  3. 타겟 자동 선택 (화면 중앙에서 가장 가까운 monster)
  4. IoU 매칭으로 같은 몬스터 추적
  5. 타겟 위치로 이동 → PRESS → 드래그 → RELEASE (공격)
  6. 사망/소실 확정(1.5초) 시 다음 타겟 자동 선택

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
from target_selector import select_nearest, select_by_click, find_matching
from death_detector import DeathDetector
from controller import PicoController, DummyController


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

        # ── 모듈 초기화 ─────────────────────────────
        dcfg = self._cfg["detector"]
        self._capture  = ScreenCapture(self._cfg["capture"]["monitor"])
        self._detector = YOLODetector(
            model_path    = dcfg["model"],
            confidence    = dcfg["confidence"],
            iou_threshold = dcfg["iou_threshold"],
            device        = dcfg["device"],
            img_size      = dcfg["img_size"],
            classes       = dcfg["classes"],
        )

        tcfg = self._cfg["target"]
        self._death_detector = DeathDetector(
            timeout_sec = tcfg["death_timeout_sec"],
            iou_thresh  = tcfg["death_iou_thresh"],
        )

        # ── 컨트롤러 (피코 HID) ─────────────────────
        ccfg = self._cfg["controller"]
        if ccfg["enabled"] and ccfg["type"] == "pico":
            self._ctrl = PicoController(port=ccfg["port"], baudrate=ccfg["baudrate"])
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
        self._drag_dx         = acfg["drag_dx"]      # 드래그 방향 X (픽셀)
        self._drag_dy         = acfg["drag_dy"]      # 드래그 방향 Y (픽셀)
        self._drag_hold_ms    = acfg["drag_hold_ms"] # PRESS 유지 시간 (ms)
        self._aim_offset_y    = acfg.get("aim_offset_y", -20)  # Y 오프셋 (위쪽 노림)
        self._last_attack_time = 0.0

        # ── 상태 변수 ───────────────────────────────
        self._target: Optional[Detection] = None
        self._target_locked = False   # True = 사망 확정 전까지 타겟 교체 금지
        self._running = False

        # ── FPS 제한 ────────────────────────────────
        self._cap_fps = self._cfg["capture"]["fps"]
        self._frame_interval = 1.0 / self._cap_fps

        # ── 모니터 절대좌표 오프셋 ──────────────────
        mon = self._capture._monitor
        self._mon_left = mon["left"]   # 0 (모니터 1)
        self._mon_top  = mon["top"]    # 0 (모니터 1)
        print(f"[Capture] 모니터 오프셋: left={self._mon_left}, top={self._mon_top}")

        # ── ROI ─────────────────────────────────────
        rcfg = self._cfg["roi"]
        if rcfg["width"] > 0 and rcfg["height"] > 0:
            self._capture.set_roi(rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])

    # ─────────────────────────────────────────────
    def _update_target(self, monsters: list):
        """
        매 프레임 타겟 갱신.
        - 타겟 없음 → 화면 중앙에서 가장 가까운 monster 자동 선택
        - 타겟 있음 → IoU 매칭으로 동일 몬스터 추적
        - 소실 1.5초 → 사망 확정 → 다음 타겟 선택
        """
        frame_h, frame_w = self._last_frame_size

        if self._target is None:
            # 타겟 없음 → 자동 선택
            if monsters:
                self._target = select_nearest(monsters, frame_w // 2, frame_h // 2)
                if self._target:
                    self._target_locked = True
                    self._death_detector.reset()
                    print(f"[Target] 자동선택: ({self._target.cx},{self._target.cy}) "
                          f"conf={self._target.confidence:.2f}")
            return

        # 타겟 있음 → IoU 매칭
        matched = find_matching(monsters, self._target,
                                self._cfg["target"]["death_iou_thresh"])
        if matched:
            self._target = matched
            self._death_detector.reset()
        else:
            # 소실 중 → 사망 판정
            dead = self._death_detector.update(monsters, self._target)
            if dead:
                print(f"[Target] 사망/소실 확정 → 다음 타겟 탐색")
                self._target = None
                self._target_locked = False
                # 즉시 다음 타겟 선택
                if monsters:
                    self._target = select_nearest(monsters, frame_w // 2, frame_h // 2)
                    if self._target:
                        self._target_locked = True
                        self._death_detector.reset()
                        print(f"[Target] 다음 타겟: ({self._target.cx},{self._target.cy})")

    # ─────────────────────────────────────────────
    def _try_attack(self):
        """
        타겟이 있고 쿨다운이 지났으면 드래그 공격 실행.
        
        공격 방식: PRESS → (drag_dx, drag_dy) MOVE → RELEASE
        몬스터 중앙 위쪽(aim_offset_y)을 노려서 히트율 향상.
        """
        if not self._attack_enabled:
            return
        if self._target is None:
            return

        now = time.time()
        if now - self._last_attack_time < self._attack_cooldown:
            return

        # 프레임 내 좌표 → Windows 절대좌표
        x = self._target.cx + self._mon_left
        y = self._target.cy + self._mon_top + self._aim_offset_y

        print(f"[Attack] 드래그공격: 프레임({self._target.cx},{self._target.cy}) "
              f"→ 절대({x},{y})  drag=({self._drag_dx},{self._drag_dy})")

        self._ctrl.drag_attack(
            x        = x,
            y        = y,
            drag_dx  = self._drag_dx,
            drag_dy  = self._drag_dy,
            hold_ms  = self._drag_hold_ms,
        )
        self._last_attack_time = now

    # ─────────────────────────────────────────────
    def _setup_hotkeys(self):
        """F9=공격토글, F10=종료 글로벌 핫키 등록."""
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
        state = "ON" if self._attack_enabled else "OFF"
        print(f"[핫키] 공격 {state}")

    def _request_stop(self):
        self._running = False
        print("[핫키] F10 → 종료 요청")

    # ─────────────────────────────────────────────
    def run(self):
        show_window = self._cfg.get("display", {}).get("show_window", False)
        self._running = True
        self._last_frame_size = (1080, 1920)

        if show_window:
            import cv2
            import visualizer
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

                # ── 탐지 ────────────────────────────────
                detections = self._detector.detect(frame)

                # monster 클래스(class_id==0)만 필터링 (adena 제외)
                monsters = [d for d in detections if d.class_id == 0]

                # 탐지 현황 로그 (5초마다)
                if not hasattr(self, "_last_detect_log"):
                    self._last_detect_log = 0.0
                if time.time() - self._last_detect_log > 5.0:
                    print(f"[Detect] monster={len(monsters)}마리 | "
                          f"전체탐지={len(detections)} | "
                          f"타겟={'있음('+str(self._target.cx)+','+str(self._target.cy)+')' if self._target else '없음'} | "
                          f"공격={'ON' if self._attack_enabled else 'OFF'}")
                    self._last_detect_log = time.time()

                # ── 타겟 갱신 ───────────────────────────
                self._update_target(monsters)

                # ── 공격 ────────────────────────────────
                self._try_attack()

                # ── 창 모드 시각화 ───────────────────────
                if show_window:
                    import cv2
                    import visualizer
                    miss = self._death_detector.miss_elapsed
                    out = visualizer.draw(
                        frame        = frame,
                        detections   = detections,
                        target       = self._target,
                        miss_elapsed = miss,
                        detector_fps = self._detector.fps,
                        capture_fps  = self._capture.fps,
                    )
                    cv2.imshow(self.WIN, out)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("c"):
                        self._target = None
                        self._target_locked = False
                        self._death_detector.reset()
                        print("[Target] 해제")

        except KeyboardInterrupt:
            print("\n[MonsterTracker] Ctrl+C → 종료")
        finally:
            if show_window:
                import cv2
                cv2.destroyAllWindows()
            self._ctrl.stop()         # 버튼 강제 해제
            self._ctrl.disconnect()
            print("[MonsterTracker] 종료 완료")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = MonsterTrackerApp()
    app.run()
