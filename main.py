"""
main.py
-------
Monster Tracker - 단일 타겟 고정 추적 시스템

실행 흐름:
  1. 화면 캡처
  2. YOLOv8으로 몬스터 탐지
  3. ByteTracker로 모든 몬스터 ID 추적
  4. 클릭으로 타겟 지정 → TargetTracker가 해당 ID 고정 추적
  5. 타겟 소실 시 칼만 예측 + Re-ID로 복구
  6. 경로/정보 시각화 + 데이터 로깅

키:
  ESC/Q : 종료
  C     : 타겟 해제
  S     : 로그 저장
  R     : ROI 재설정
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
from detector import YOLODetector
from target_tracker import TargetTracker, TARGET_NONE
from data_logger import DataLogger
from visualizer import Visualizer


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MonsterTrackerApp:

    def __init__(self):
        self._cfg = load_config(
            os.path.join(os.path.dirname(__file__), "config.json"))

        # 모듈 초기화
        self._capture    = ScreenCapture(self._cfg["capture"]["monitor"])
        self._detector   = YOLODetector(
            model_path  = self._cfg["detector"]["model"],
            confidence  = self._cfg["detector"]["confidence"],
            iou_threshold= self._cfg["detector"]["iou_threshold"],
            device      = self._cfg["detector"]["device"],
            img_size    = self._cfg["detector"]["img_size"],
            classes     = self._cfg["detector"]["classes"],
        )
        self._tracker    = TargetTracker(self._cfg)
        self._logger     = DataLogger(
            save_dir    = self._cfg["logger"]["save_dir"],
            max_records = self._cfg["logger"]["max_records"],
        )
        self._viz        = Visualizer()

        # 클릭 상태
        self._click_pos: Optional[tuple] = None
        self._all_tracks = []

        # FPS
        self._cap_fps_ticks = []
        self._cap_fps = 0.0
        self._target_interval = 1.0 / self._cfg["capture"]["fps"]

    # ------------------------------------------------------------------
    # ROI 설정
    # ------------------------------------------------------------------

    def setup(self):
        print("\n" + "="*50)
        print("  Monster Tracker v1.0")
        print("  클릭으로 추적할 몬스터를 지정하세요")
        print("="*50)
        print("\n[Setup] ROI 설정 (Enter=전체화면, n=새로지정)")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice == "n":
            frame = self._capture.capture_full()
            if frame is not None:
                result = self._capture.select_roi_interactive(frame)
                if result:
                    x, y, w, h = result
                    self._capture.set_roi(x, y, w, h)
                    print(f"[Setup] ROI 설정: ({x},{y}) {w}x{h}")
                else:
                    print("[Setup] ROI 취소 → 전체화면 사용")
        else:
            print("[Setup] 전체화면 사용")

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------

    def run(self):
        win = "Monster Tracker (클릭=타겟지정)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_callback)

        print("\n[Run] 시작! 몬스터를 클릭해서 추적하세요.")
        print("  ESC/Q: 종료 | C: 타겟해제 | S: 로그저장 | R: ROI재설정\n")

        frame_count = 0

        try:
            while True:
                loop_start = time.time()

                # ── 1. 캡처 ───────────────────────────────────────────
                frame = self._capture.capture()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # ── 2. 탐지 ───────────────────────────────────────────
                detections = self._detector.detect(frame)

                # ── 3. 추적 업데이트 ──────────────────────────────────
                self._all_tracks, target_info = self._tracker.update(
                    detections, frame)

                # ── 4. 클릭으로 타겟 지정 ─────────────────────────────
                if self._click_pos is not None:
                    self._handle_click(self._click_pos, frame)
                    self._click_pos = None

                # ── 5. 로깅 ───────────────────────────────────────────
                if self._cfg["logger"]["enabled"]:
                    self._logger.log(target_info)

                # ── 6. 경로 포인트 ────────────────────────────────────
                path_points = self._logger.get_path_points(last_n=80)

                # ── 7. 시각화 ─────────────────────────────────────────
                self._tick_fps()
                out = self._viz.draw(
                    frame        = frame,
                    tracks       = self._all_tracks,
                    target_info  = target_info,
                    path_points  = path_points,
                    snapshot     = self._tracker.snapshot,
                    capture_fps  = self._cap_fps,
                    detect_fps   = self._detector.fps,
                )
                cv2.imshow(win, out)

                # ── 8. 키 입력 ────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    print("[Main] 종료")
                    break
                elif key == ord('c'):
                    self._tracker.clear_target()
                    self._logger.clear()
                elif key == ord('s'):
                    self._logger.save_csv()
                elif key == ord('r'):
                    frame_full = self._capture.capture_full()
                    if frame_full is not None:
                        result = self._capture.select_roi_interactive(frame_full)
                        if result:
                            x, y, w, h = result
                            self._capture.set_roi(x, y, w, h)

                # ── 9. FPS 제한 ───────────────────────────────────────
                elapsed = time.time() - loop_start
                sleep_t = self._target_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

                frame_count += 1

        except KeyboardInterrupt:
            print("\n[Main] 중단")
        finally:
            if self._cfg["logger"]["enabled"] and self._logger.record_count > 0:
                self._logger.save_csv()
            cv2.destroyAllWindows()
            print("[Main] 종료 완료")

    # ------------------------------------------------------------------
    # 클릭 핸들러
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_pos = (x, y)

    def _handle_click(self, pos, frame):
        """클릭 위치에서 가장 가까운 트랙을 타겟으로 지정."""
        cx, cy = pos
        best_track = None
        best_dist = float('inf')

        for t in self._all_tracks:
            dist = np.sqrt((t.cx - cx)**2 + (t.cy - cy)**2)
            # 박스 안을 클릭했는지 확인
            x, y, w, h = t.predicted_bbox
            in_box = (x <= cx <= x+w) and (y <= cy <= y+h)
            score = dist if not in_box else dist * 0.1
            if score < best_dist:
                best_dist = score
                best_track = t

        if best_track and best_dist < 200:
            self._tracker.set_target(
                best_track.track_id, self._all_tracks, frame)
            self._logger.clear()
            print(f"[Main] 클릭 타겟 지정: #{best_track.track_id} "
                  f"({best_track.cx}, {best_track.cy})")
        else:
            print(f"[Main] 클릭 위치 ({cx},{cy}) 근처에 트랙 없음")

    def _tick_fps(self):
        now = time.time()
        self._cap_fps_ticks.append(now)
        if len(self._cap_fps_ticks) > 30:
            self._cap_fps_ticks.pop(0)
        if len(self._cap_fps_ticks) >= 2:
            e = self._cap_fps_ticks[-1] - self._cap_fps_ticks[0]
            if e > 0:
                self._cap_fps = round((len(self._cap_fps_ticks)-1)/e, 1)


# ------------------------------------------------------------------
# 엔트리 포인트
# ------------------------------------------------------------------

def main():
    app = MonsterTrackerApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
