"""
main.py
-------
Monster Tracker - 재설계판
죽고 새로 스폰되는 몬스터 구조에 맞춘 단순 타겟 추적.

흐름:
  1. 화면 캡처
  2. YOLOv8s 탐지
  3. 타겟 지정 (클릭) → 매 프레임 IoU 매칭으로 동일 몬스터 유지
  4. 타겟 사망/소실 감지 → 자동으로 가장 가까운 새 몬스터 선택
  5. 시각화

키:
  마우스 클릭 : 타겟 지정
  C           : 타겟 해제
  R           : ROI 재설정
  Q / ESC     : 종료
"""

import sys
import os
import cv2
import json
import time
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection
from target_selector import select_nearest, select_by_click, find_matching
from death_detector import DeathDetector
import visualizer

# controller.py는 monster_bot 것을 그대로 사용
# make_controller는 config dict 기반으로 직접 생성
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
            model_path   = dcfg["model"],
            confidence   = dcfg["confidence"],
            iou_threshold= dcfg["iou_threshold"],
            device       = dcfg["device"],
            img_size     = dcfg["img_size"],
            classes      = dcfg["classes"],
        )

        tcfg = self._cfg["target"]
        self._death_detector = DeathDetector(
            timeout_sec = tcfg["death_timeout_sec"],
            iou_thresh  = tcfg["death_iou_thresh"],
        )
        self._select_mode  = tcfg["select_mode"]
        self._click_radius = tcfg["click_radius"]

        # ── 컨트롤러 (피코 HID) ─────────────────────
        ccfg = self._cfg["controller"]
        if ccfg["enabled"] and ccfg["type"] == "pico":
            self._controller = PicoController(
                port=ccfg["port"], baudrate=ccfg["baudrate"])
            if not self._controller.connect():
                print("[Controller] 피코 연결 실패 → Dummy 모드")
                self._controller = DummyController()
                self._controller.connect()
        else:
            self._controller = DummyController()
            self._controller.connect()

        # ── 공격 설정 ───────────────────────────────
        acfg = self._cfg["attack"]
        self._attack_enabled  = acfg["enabled"]
        self._attack_cooldown = acfg["cooldown_sec"]
        self._drag_dx         = acfg["drag_dx"]
        self._drag_dy         = acfg["drag_dy"]
        self._drag_hold_ms    = acfg["drag_hold_ms"]
        self._aim_offset_y    = acfg.get("aim_offset_y", -20)  # 클릭 Y 오프셋 (위쪽)
        self._last_attack_time = 0.0

        # ── 상태 변수 ───────────────────────────────
        self._target: Optional[Detection] = None
        self._target_locked = False   # True = 타겟 고정 중 (사망 전까지 교체 안 함)
        self._auto_select = True
        self._pending_click: Optional[tuple] = None

        # ── FPS 캡처 제한 ────────────────────────────
        self._cap_fps   = self._cfg["capture"]["fps"]
        self._frame_interval = 1.0 / self._cap_fps

        # ── 모니터 절대좌표 오프셋 (프레임 내 좌표 → Windows 절대좌표 변환용) ──
        mon = self._capture._monitor   # mss monitor dict: left, top, width, height
        self._mon_left = mon["left"]
        self._mon_top  = mon["top"]
        print(f"[Capture] 모니터 오프셋: left={self._mon_left}, top={self._mon_top}")

        # ── ROI ─────────────────────────────────────
        rcfg = self._cfg["roi"]
        if rcfg["width"] > 0 and rcfg["height"] > 0:
            self._capture.set_roi(
                rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])

    # ─────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pending_click = (x, y)

    # ─────────────────────────────────────────────
    def _handle_click(self, x: int, y: int,
                      detections: list) -> Optional[Detection]:
        """클릭 위치로 타겟 지정."""
        picked = select_by_click(detections, x, y, self._click_radius)
        if picked is None:
            # 반경 내 몬스터 없으면 가장 가까운 것 선택
            frame_h, frame_w = self._last_frame_size
            picked = select_nearest(detections, x, y)
        if picked:
            self._death_detector.reset()
            print(f"[Target] 지정: ({picked.cx},{picked.cy}) "
                  f"conf={picked.confidence:.2f}")
        return picked

    # ─────────────────────────────────────────────
    def _update_target(self, detections: list):
        """매 프레임 타겟 갱신."""

        # monster 클래스만 필터링 (adena 제외)
        monsters = [d for d in detections if d.class_id == 0]

        # 1) 클릭 처리
        if self._pending_click is not None:
            cx, cy = self._pending_click
            self._pending_click = None
            self._auto_select = False
            self._target = self._handle_click(cx, cy, monsters)
            if self._target:
                self._target_locked = True
            return

        # 2) 타겟 없음 → 자동 선택 (최초 1회, 이후 사망 확정 시에만)
        if self._target is None:
            self._target_locked = False
            if self._auto_select and monsters:
                frame_h, frame_w = self._last_frame_size
                self._target = select_nearest(
                    monsters, frame_w // 2, frame_h // 2)
                if self._target:
                    self._target_locked = True
                    self._death_detector.reset()
                    print(f"[Target] 자동선택: ({self._target.cx},{self._target.cy})")
            return

        # 3) 타겟 고정 중 → IoU 매칭으로 동일 몬스터만 추적
        matched = find_matching(monsters, self._target,
                                self._cfg["target"]["death_iou_thresh"])
        if matched:
            self._target = matched  # 최신 박스로 업데이트 (같은 몬스터)
            self._death_detector.reset()
        else:
            # 소실 중 → DeathDetector 판정 (다른 몬스터로 절대 교체 안 함)
            dead = self._death_detector.update(monsters, self._target)
            if dead:
                print("[Target] 사망/소실 확정 → 다음 타겟 탐색")
                self._target = None
                self._target_locked = False
                # 사망 확정 후에만 다음 nearest 선택
                if self._auto_select and monsters:
                    frame_h, frame_w = self._last_frame_size
                    self._target = select_nearest(
                        monsters, frame_w // 2, frame_h // 2)
                    if self._target:
                        self._target_locked = True
                        self._death_detector.reset()
                        print(f"[Target] 다음 타겟: ({self._target.cx},{self._target.cy})")

    # ─────────────────────────────────────────────
    def _try_attack(self):
        """타겟이 있으면 쿨다운 체크 후 공격."""
        if not self._attack_enabled:
            return
        if self._target is None:
            return
        now = time.time()
        if now - self._last_attack_time < self._attack_cooldown:
            return

        # 프레임 내 상대좌표 → Windows 절대좌표 변환
        # aim_offset_y: 몬스터 중앙보다 위쪽(상체)을 노림 (기본 -20px)
        x = self._target.cx + self._mon_left
        y = self._target.cy + self._mon_top + self._aim_offset_y
        self._controller.click_drag(
            x, y,
            drag_dx = self._drag_dx,
            drag_dy = self._drag_dy,
            hold_ms = self._drag_hold_ms,
        )
        self._last_attack_time = now
        print(f"[Attack] 공격: 프레임({self._target.cx},{self._target.cy}) → 절대({x},{y})")

    # ─────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN, self._on_mouse)

        self._last_frame_size = (720, 1280)  # 기본값, 첫 프레임에서 갱신

        print("[MonsterTracker] 시작. Q/ESC=종료, Click=타겟지정, C=해제, R=ROI재설정")

        last_time = time.time()

        while True:
            # ── FPS 제한 ──────────────────────────────
            now = time.time()
            elapsed = now - last_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
            last_time = time.time()

            # ── 캡처 ──────────────────────────────────
            frame = self._capture.capture()
            if frame is None:
                continue
            self._last_frame_size = (frame.shape[0], frame.shape[1])

            # ── 탐지 ──────────────────────────────────
            detections = self._detector.detect(frame)

            # ── 타겟 갱신 ─────────────────────────────
            self._update_target(detections)

            # ── 공격 ──────────────────────────────────
            self._try_attack()

            # ── 시각화 ────────────────────────────────
            miss = self._death_detector.miss_elapsed
            out = visualizer.draw(
                frame       = frame,
                detections  = detections,
                target      = self._target,
                miss_elapsed= miss,
                detector_fps= self._detector.fps,
                capture_fps = self._capture.fps,
            )
            cv2.imshow(self.WIN, out)

            # ── 키 입력 ───────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # Q or ESC
                break

            elif key == ord('c'):      # C: 타겟 해제
                self._target = None
                self._auto_select = False
                self._death_detector.reset()
                print("[Target] 해제")

            elif key == ord('r'):      # R: ROI 재설정
                full = self._capture.capture_full()
                if full is not None:
                    result = self._capture.select_roi_interactive(full)
                    if result:
                        x, y, w, h = result
                        self._capture.set_roi(x, y, w, h)
                    else:
                        self._capture.clear_roi()
                        print("[Capture] ROI 해제 (전체 화면)")

        cv2.destroyAllWindows()
        self._controller.disconnect()
        print("[MonsterTracker] 종료")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = MonsterTrackerApp()
    app.run()
