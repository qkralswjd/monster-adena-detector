"""
overlay_detect.py
-----------------
[단계 3] YOLO 탐지 연결 + 오버레이 시각화.

detect_test.py 의 오버레이 버전.
OpenCV 별도 창 대신 게임화면 위에 직접 박스를 그림.

목적:
  - 화면캡처 → YOLO → OverlayWindow 로 바운딩박스 표시
  - 자동 클릭 없음 (탐지 정확도 확인 전용)
  - 오버레이 클릭 시 PICO 전달은 ctrl=None 이므로 무시됨

확인 항목:
  1. 바운딩박스가 실제 몬스터 위에 정확히 표시되는지
  2. cx,cy 가 몬스터 중앙인지
  3. 로그 좌표와 화면 좌표가 일치하는지

종료: Ctrl+C (콘솔에서)

사용법:
  python overlay_detect.py
"""

import sys
import os
import time
import json
import threading

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector
from target_selector import select_by_confidence
from overlay_window import OverlayWindow


def main():
    # ── config 로드 ───────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # ── 캡처 초기화 ───────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"])
    print(f"[Init] 캡처: {cap.width}x{cap.height}  "
          f"left={cap.left} top={cap.top}")

    # ── YOLO 초기화 ───────────────────────────────────────────
    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
    )

    # ── 오버레이 초기화 ───────────────────────────────────────
    # lb_x=0: 레터박스 없음 (1920x1080 풀스크린)
    # ctrl=None: 탐지 확인 전용, 자동 클릭 없음
    ov = OverlayWindow(
        mon_left = cap.left,
        mon_top  = cap.top,
        game_w   = cap.width,
        game_h   = cap.height,
        lb_x     = 0,
        ctrl     = None,
    )
    ov.start()
    print(f"[Init] 오버레이: {cap.width}x{cap.height} "
          f"위치=({cap.left},{cap.top})")
    print()
    print("[Start] 탐지 시작. Ctrl+C 로 종료.")
    print("        바운딩박스가 실제 몬스터 위에 정확히 표시되는지 확인하세요.")
    print()

    # ── 로그용 타이머 ─────────────────────────────────────────
    prev_log_t  = 0.0
    LOG_INTERVAL = 0.5

    try:
        while True:
            # ── 캡처 ──────────────────────────────────────────
            frame = cap.capture()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── YOLO 탐지 ─────────────────────────────────────
            detections = det.detect(frame)

            monsters = [d for d in detections if d.class_id == 0]
            adenas   = [d for d in detections if d.class_id == 1]

            # ── 타겟 선택 ─────────────────────────────────────
            target = select_by_confidence(detections)

            # ── 오버레이 갱신 ─────────────────────────────────
            ov.update(
                detections   = detections,
                target       = target,
                miss_elapsed = 0.0,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
            )

            # ── 로그 출력 (0.5초마다) ─────────────────────────
            now = time.time()
            if now - prev_log_t >= LOG_INTERVAL:
                prev_log_t = now
                if detections:
                    print(f"── 탐지 {len(monsters)}마리  "
                          f"아데나 {len(adenas)}  "
                          f"DET {det.fps}fps ──")
                    for d in monsters:
                        mark = " ← TARGET" if (
                            target and
                            d.cx == target.cx and d.cy == target.cy
                        ) else ""
                        print(f"  [monster] "
                              f"x={d.x} y={d.y} "
                              f"w={d.w} h={d.h} "
                              f"cx={d.cx} cy={d.cy} "
                              f"conf={d.confidence:.2f}{mark}")
                    for d in adenas:
                        print(f"  [adena]   "
                              f"x={d.x} y={d.y} "
                              f"cx={d.cx} cy={d.cy} "
                              f"conf={d.confidence:.2f}")
                else:
                    print(f"  탐지 없음  DET {det.fps:.1f}fps")

            time.sleep(0.01)  # CPU 과부하 방지

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        ov.stop()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
