"""
main.py
-------
Monster Tracker - 탐지 좌표 로그 버전

전체 흐름:
  1. 화면 캡처 (mss monitors[1], left=-1920)
  2. YOLOv8s 탐지 (monster=0 / adena=1)
  3. ROI 필터링
  4. ROI 안 몬스터 전체 목록 로그 출력
  5. 탐지 순서대로 타겟 큐 구성 (1번→2번→...)
  6. 현재 타겟 추적 - 매 프레임 전체화면 절대좌표 출력
  7. 사망/소실(1.5초) → 다음 타겟으로

좌표 기준:
  프레임좌표  = mss 캡처 기준 (0~1920, 0~1080)
  전체화면좌표 = 프레임좌표 + mon_left/top
                 게임이 left=-1920 이면 전체화면x = 프레임x - 1920

핫키:
  Q / ESC : 종료 (창 모드)
  Ctrl+C  : 강제종료
"""

import sys
import os
import time
import json
from typing import Optional, List

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection
from target_selector import find_matching
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

        # ── 모니터 정보 ─────────────────────────────
        mon = self._capture._monitor
        self._mon_left = mon["left"]   # 게임모니터 = -1920
        self._mon_top  = mon["top"]    # 보통 0
        mss_w = mon["width"]
        mss_h = mon["height"]
        print(f"[Capture] 게임모니터: left={self._mon_left}, top={self._mon_top} "
              f"| 캡처 크기: {mss_w}x{mss_h}")
        print(f"[Capture] 전체화면좌표 = 프레임좌표 + ({self._mon_left},{self._mon_top})")

        # ── ROI (탐지 필터 존) ──────────────────────
        rcfg = self._cfg["roi"]
        if rcfg["width"] > 0 and rcfg["height"] > 0:
            self._roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])
            print(f"[ROI] 탐지존: 프레임({rcfg['x']},{rcfg['y']}) "
                  f"크기 {rcfg['width']}x{rcfg['height']}")
        else:
            self._roi = None
            print("[ROI] 전체 화면 탐지")

        # ── 타겟 상태 ───────────────────────────────
        self._target: Optional[Detection] = None
        self._target_no: int = 0          # 현재 타겟 번호
        # 이전 프레임 몬스터 ID 목록 (새 몬스터 감지용)
        self._known_monster_ids: List[tuple] = []  # (cx, cy) 기준
        self._running = False

        # ── FPS 제한 ────────────────────────────────
        self._cap_fps        = self._cfg["capture"]["fps"]
        self._frame_interval = 1.0 / self._cap_fps

        # ── 로그 타이머 ─────────────────────────────
        self._last_status_time = 0.0
        self._last_coord_time  = 0.0
        self._last_monsters_log: List[Detection] = []   # 직전 프레임 몬스터 목록

    # ── 전체화면 절대좌표 변환 ────────────────────────
    def _to_screen(self, frame_x: int, frame_y: int):
        return frame_x + self._mon_left, frame_y + self._mon_top

    # ══════════════════════════════════════════════
    #  몬스터 목록 로그 (변화 있을 때만 출력)
    # ══════════════════════════════════════════════
    def _log_monsters(self, monsters: List[Detection]):
        """
        ROI 안 몬스터 전체 목록 출력.
        탐지된 몬스터 수/위치가 바뀔 때만 출력 (매 프레임 X).
        """
        # 현재 프레임 cx/cy 집합
        cur_ids = [(m.cx, m.cy) for m in monsters]
        if cur_ids == self._last_monsters_log:
            return
        self._last_monsters_log = cur_ids

        if not monsters:
            print(f"[탐지] 몬스터 없음")
            return

        print(f"[탐지] 몬스터 {len(monsters)}마리 ─────────────────")
        for i, m in enumerate(monsters, 1):
            sc_x, sc_y = self._to_screen(m.cx, m.cy)
            target_mark = " ◀ 현재타겟" if (self._target and
                          m.cx == self._target.cx and
                          m.cy == self._target.cy) else ""
            print(f"  [{i}번] 프레임({m.cx:4d},{m.cy:4d})  "
                  f"전체화면({sc_x:5d},{sc_y:4d})  "
                  f"conf={m.confidence:.2f}  "
                  f"bbox=({m.x},{m.y},{m.w},{m.h}){target_mark}")

    # ══════════════════════════════════════════════
    #  타겟 갱신 (탐지 순서대로 1→2→3...)
    # ══════════════════════════════════════════════
    def _update_target(self, monsters: List[Detection]) -> bool:
        """
        타겟 없음 → 탐지 순서 1번(리스트 첫 번째)을 타겟으로 지정.
        타겟 있음 → IoU 매칭으로 추적.
        소실 1.5초 → 사망 확정 → True 반환.
        """
        if self._target is None:
            if monsters:
                # 탐지 순서 첫 번째를 타겟으로
                self._target = monsters[0]
                self._target_no += 1
                self._death_detector.reset()
                sc_x, sc_y = self._to_screen(self._target.cx, self._target.cy)
                print(f"[Target #{self._target_no}] 선택: "
                      f"프레임({self._target.cx},{self._target.cy})  "
                      f"전체화면({sc_x},{sc_y})  "
                      f"conf={self._target.confidence:.2f}")
            return False

        # IoU 매칭으로 같은 몬스터 추적
        matched = find_matching(monsters, self._target,
                                self._cfg["target"]["death_iou_thresh"])
        if matched:
            self._target = matched
            self._death_detector.reset()
            return False

        # 소실 → 사망 판정
        dead = self._death_detector.update(monsters, self._target)
        if dead:
            sc_x, sc_y = self._to_screen(self._target.cx, self._target.cy)
            print(f"[Target #{self._target_no}] 사망/소실 확정! "
                  f"마지막위치: 프레임({self._target.cx},{self._target.cy})  "
                  f"전체화면({sc_x},{sc_y})")
            return True

        return False

    # ══════════════════════════════════════════════
    #  현재 타겟 이동 좌표 로그 (매 프레임)
    # ══════════════════════════════════════════════
    def _log_target_coord(self):
        """현재 타겟 좌표를 매 프레임 출력"""
        if self._target is None:
            return
        now = time.time()
        if now - self._last_coord_time < 0.1:   # 0.1초마다 (10fps)
            return
        self._last_coord_time = now

        sc_x, sc_y = self._to_screen(self._target.cx, self._target.cy)
        miss = self._death_detector.miss_elapsed
        miss_str = f"  소실중={miss:.1f}s" if miss > 0 else ""
        print(f"[타겟 #{self._target_no}] "
              f"프레임({self._target.cx:4d},{self._target.cy:4d})  "
              f"전체화면({sc_x:5d},{sc_y:4d})  "
              f"conf={self._target.confidence:.2f}{miss_str}")

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
            print("[MonsterTracker] 창 모드. Q/ESC=종료  C=타겟해제")
        else:
            print("[MonsterTracker] 백그라운드 모드. Ctrl+C=종료")

        print("=" * 60)
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

                # ── 몬스터 전체 목록 로그 ─────────────
                self._log_monsters(monsters)

                # ── 타겟 갱신 ─────────────────────────
                dead = self._update_target(monsters)
                if dead:
                    self._target = None

                # ── 현재 타겟 이동 좌표 출력 ──────────
                self._log_target_coord()

                # ── 5초마다 상태 요약 ─────────────────
                if time.time() - self._last_status_time > 5.0:
                    target_str = (f"#{self._target_no} 프레임({self._target.cx},{self._target.cy})"
                                  if self._target else "없음")
                    print(f"[Status] monster={len(monsters)} | "
                          f"adena={len(adenas)} | "
                          f"타겟={target_str} | "
                          f"탐지FPS={self._detector.fps}")
                    self._last_status_time = time.time()

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
                self._last_monsters_log = []
                print("[Target] 수동 해제")
        except Exception as e:
            print(f"[Render] 오류: {e}")


if __name__ == "__main__":
    app = MonsterTrackerApp()
    app.run()
