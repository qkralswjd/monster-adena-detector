"""
overlay_detect.py
-----------------
[단계 3] YOLO 탐지 연결 + 오버레이 시각화.

목적:
  - 화면캡처 → YOLO → OverlayWindow 로 바운딩박스 표시
  - 자동 클릭 없음 (탐지 정확도 확인 전용)
  - miss_timeout: 탐지 없어도 마지막 결과를 잠깐 유지 (깜빡임 방지)

종료: Ctrl+C

사용법:
  python overlay_detect.py
"""

import sys
import os
import time
import json

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

    miss_timeout = cfg["target"].get("miss_timeout_sec", 1.5)

    # ── 캡처 초기화 ───────────────────────────────────────────
    cap = ScreenCapture(monitor=cfg["capture"]["monitor"])
    print(f"[Init] 캡처: {cap.width}x{cap.height}  left={cap.left} top={cap.top}")

    # ── YOLO 초기화 ───────────────────────────────────────────
    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
    )
    print(f"[Init] YOLO conf={dcfg['confidence']}  miss_timeout={miss_timeout}s")

    # ── 오버레이 초기화 ───────────────────────────────────────
    ov = OverlayWindow(
        mon_left = cap.left,
        mon_top  = cap.top,
        game_w   = cap.width,
        game_h   = cap.height,
        lb_x     = 0,       # 레터박스 없음
        ctrl     = None,    # 탐지 확인 전용 - 클릭 없음
    )
    ov.start()
    print(f"[Init] 오버레이 시작  {cap.width}x{cap.height} @ ({cap.left},{cap.top})")
    print()
    print("[Start] 탐지 시작. Ctrl+C 로 종료.")
    print()

    # ── miss_timeout 버퍼 ────────────────────────────────────
    # 탐지 없어도 마지막 결과를 miss_timeout 초 동안 유지 → 깜빡임 방지
    last_detections  = []
    last_target      = None
    last_det_time    = 0.0   # 마지막으로 탐지된 시각

    # ── 로그 타이머 ───────────────────────────────────────────
    prev_log_t   = 0.0
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
            now = time.time()

            monsters = [d for d in detections if d.class_id == 0]
            adenas   = [d for d in detections if d.class_id == 1]

            if detections:
                # 새 탐지 결과 업데이트
                last_detections = detections
                last_target     = select_by_confidence(detections)
                last_det_time   = now
                miss_elapsed    = 0.0
            else:
                # 탐지 없음 → miss_timeout 동안 마지막 결과 유지
                miss_elapsed = now - last_det_time
                if miss_elapsed > miss_timeout:
                    # timeout 초과 → 화면 지우기
                    last_detections = []
                    last_target     = None

            # ── 오버레이 갱신 ─────────────────────────────────
            ov.update(
                detections   = last_detections,
                target       = last_target,
                miss_elapsed = miss_elapsed if not detections else 0.0,
                det_fps      = det.fps,
                cap_fps      = cap.fps,
            )

            # ── 로그 출력 (0.5초마다) ─────────────────────────
            if now - prev_log_t >= LOG_INTERVAL:
                prev_log_t = now
                if monsters or adenas:
                    print(f"── 탐지 {len(monsters)}마리  아데나 {len(adenas)}  "
                          f"DET {det.fps:.1f}fps ──")
                    for d in monsters:
                        mark = " ← TARGET" if (
                            last_target and
                            d.cx == last_target.cx and d.cy == last_target.cy
                        ) else ""
                        print(f"  [monster] cx={d.cx} cy={d.cy}  "
                              f"w={d.w} h={d.h}  conf={d.confidence:.2f}{mark}")
                    for d in adenas:
                        print(f"  [adena]   cx={d.cx} cy={d.cy}  "
                              f"conf={d.confidence:.2f}")
                else:
                    miss_sec = now - last_det_time
                    if last_detections:
                        print(f"  탐지 없음 (마지막 결과 유지 중 {miss_sec:.1f}s)  "
                              f"DET {det.fps:.1f}fps")
                    else:
                        print(f"  탐지 없음  DET {det.fps:.1f}fps")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        ov.stop()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
