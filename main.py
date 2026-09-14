"""
main.py
-------
Monster Tracker - 탐지 + 좌표 로그 버전 (피코 제거)

전체 흐름:
  1. 화면 캡처 (mss, 모니터 1)
  2. YOLOv8s 탐지 (monster / adena 2클래스)
  3. ROI 필터링
  4. 타겟 자동 선택 (중앙에서 가장 가까운 monster)
  5. IoU 매칭으로 같은 몬스터 추적
  6. [좌표] 로그 출력 (프레임좌표 / bbox / conf / 화면비율)
  7. 사망/소실 확정(1.5초) → 아데나 수집 단계
  8. 다음 몬스터 선택

핫키:
  Q / ESC : 종료 (창 모드)
  Ctrl+C  : 강제종료
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


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            classes       = dcfg["classes"],
        )

        # ── 사망 감지기 ─────────────────────────────
        tcfg = self._cfg["target"]
        self._death_detector = DeathDetector(
            timeout_sec = tcfg["death_timeout_sec"],
            iou_thresh  = tcfg["death_iou_thresh"],
        )

        # ── mss 실제 캡처 크기 확인 ─────────────────
        mon = self._capture._monitor
        mss_w = mon["width"]
        mss_h = mon["height"]
        self._mon_left = mon["left"]
        self._mon_top  = mon["top"]
        print(f"[Capture] 게임모니터: left={self._mon_left}, top={self._mon_top} "
              f"| 캡처 크기: {mss_w}x{mss_h}")

        # ── 상태 변수 ───────────────────────────────
        self._target: Optional[Detection] = None
        self._running = False

        # ── FPS 제한 ────────────────────────────────
        self._cap_fps        = self._cfg["capture"]["fps"]
        self._frame_interval = 1.0 / self._cap_fps

        # ── ROI (탐지 필터 존) ──────────────────────
        rcfg = self._cfg["roi"]
        if rcfg["width"] > 0 and rcfg["height"] > 0:
            self._roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])
            print(f"[ROI] 탐지존: ({rcfg['x']},{rcfg['y']}) {rcfg['width']}x{rcfg['height']}")
        else:
            self._roi = None
            print("[ROI] 전체 화면 탐지")

        # ── 로그 타이머 ─────────────────────────────
        self._last_log_time = 0.0
        self._last_coord_time = 0.0

    # ══════════════════════════════════════════════
    #  타겟 갱신
    # ══════════════════════════════════════════════
    def _update_target(self, monsters: list) -> bool:
        frame_h, frame_w = self._last_frame_size

        if self._target is None:
            if monsters:
                self._target = select_nearest(monsters, frame_w // 2, frame_h // 2)
                if self._target:
                    self._death_detector.reset()
                    print(f"[Target] 자동선택: ({self._target.cx},{self._target.cy}) "
                          f"conf={self._target.confidence:.2f}")
            return False

        matched = find_matching(monsters, self._target,
                                self._cfg["target"]["death_iou_thresh"])
        if matched:
            self._target = matched
            self._death_detector.reset()
            return False

        dead = self._death_detector.update(monsters, self._target)
        if dead:
            print(f"[Target] 사망/소실 확정!")
            return True

        return False

    # ══════════════════════════════════════════════
    #  좌표 로그
    # ══════════════════════════════════════════════
    def _log_coord(self):
        """쿨다운마다 타겟 좌표 출력"""
        if self._target is None:
            return
        now = time.time()
        if now - self._last_coord_time < 0.5:
            return
        self._last_coord_time = now

        fw, fh = self._last_frame_size[1], self._last_frame_size[0]
        cx, cy = self._target.cx, self._target.cy
        print(f"[좌표] 몬스터 프레임({cx},{cy})  "
              f"bbox=({self._target.x},{self._target.y},{self._target.w},{self._target.h})  "
              f"conf={self._target.confidence:.2f}  "
              f"화면비=({cx/fw:.3f},{cy/fh:.3f})")

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
            print("[MonsterTracker] 백그라운드 모드. Ctrl+C=종료")

        last_time = time.time()

        try:
            while self._running:
                # ── FPS 제한 ──────────────────────────
                now = time.time()
                elapsed = now - last_time
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                last_time = time.time()

                # ── 캡처 ──────────────────────────────
                frame = self._capture.capture()
                if frame is None:
                    continue
                self._last_frame_size = (frame.shape[0], frame.shape[1])

                # ── 탐지 ──────────────────────────────
                all_detections = self._detector.detect(frame)

                # ── ROI 필터링 ────────────────────────
                if self._roi:
                    rx, ry, rw, rh = self._roi
                    all_detections = [
                        d for d in all_detections
                        if rx <= d.cx <= rx + rw and ry <= d.cy <= ry + rh
                    ]

                monsters = [d for d in all_detections if d.class_id == 0]
                adenas   = [d for d in all_detections if d.class_id == 1]

                # ── 5초마다 현황 로그 ─────────────────
                if time.time() - self._last_log_time > 5.0:
                    target_str = (f"있음({self._target.cx},{self._target.cy})"
                                  if self._target else "없음")
                    print(f"[Status] monster={len(monsters)} | "
                          f"adena={len(adenas)}개 | "
                          f"타겟={target_str}")
                    self._last_log_time = time.time()

                # ── 타겟 갱신 ────────────────────────
                dead = self._update_target(monsters)
                if dead:
                    self._target = None

                # ── 좌표 로그 출력 ────────────────────
                self._log_coord()

                # ── 창 모드 시각화 ────────────────────
                if show_window:
                    self._render_window(frame, all_detections)

        except KeyboardInterrupt:
            print("\n[MonsterTracker] Ctrl+C → 종료")
        finally:
            if show_window:
                import cv2
                cv2.destroyAllWindows()
            print("[MonsterTracker] 종료 완료")

    # ══════════════════════════════════════════════
    #  창 모드 렌더링
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
                roi          = self._roi,
            )
            cv2.imshow(self.WIN, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._running = False
            elif key == ord("c"):
                self._target = None
                self._death_detector.reset()
                print("[Target] 해제")
        except Exception as e:
            print(f"[Render] 오류: {e}")


if __name__ == "__main__":
    app = MonsterTrackerApp()
    app.run()
