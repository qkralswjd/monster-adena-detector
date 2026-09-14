"""
detect_test.py
--------------
[단계 3] YOLO 탐지 연결 + 좌표 확인.

목적:
  - 화면 캡처 → YOLO → Detection cx/cy 가 실제 몬스터 위치와 일치하는지 확인
  - 자동 클릭/공격 없음
  - 탐지 결과 로그 + OpenCV 시각화 창

시각화:
  - 탐지된 몬스터: 빨간 박스 + cx/cy 표시
  - 선택된 타겟: 초록 박스 (confidence 최고)
  - 아데나: 노란 박스
  - 왼쪽 상단: FPS / 탐지 수 / 타겟 cx,cy

확인 항목:
  1. 바운딩박스가 실제 몬스터 위에 정확히 표시되는지
  2. cx,cy 가 몬스터 중앙인지
  3. 로그 좌표와 화면 좌표가 일치하는지

종료: Q 또는 ESC

사용법:
  python detect_test.py
  python detect_test.py --monitor 1   # 캡처 모니터 지정
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection
from target_selector import select_by_confidence


# ── 시각화 설정 ──────────────────────────────────────────────
COLOR_MONSTER = (0,   80, 255)   # 빨강 (BGR)
COLOR_TARGET  = (0,  220,   0)   # 초록
COLOR_ADENA   = (0,  220, 220)   # 노랑
COLOR_TEXT    = (255, 255, 255)  # 흰색
COLOR_BG      = (0,   0,   0)    # 검정 (텍스트 배경)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS  = 1


def draw_detection(img: np.ndarray, d: Detection,
                   color: tuple, label: str = ""):
    """바운딩박스 + 중심점 + 라벨 그리기."""
    x1, y1, x2, y2 = d.tlbr
    cx, cy = d.cx, d.cy

    # 박스
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 중심 십자
    cv2.drawMarker(img, (cx, cy), color,
                   cv2.MARKER_CROSS, 12, 2)

    # 라벨 (박스 위)
    text = label or f"{d.class_name} {d.confidence:.2f}"
    coord_text = f"cx={cx} cy={cy}"

    # 라벨 배경
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                FONT, FONT_SCALE, COLOR_TEXT, THICKNESS)

    # cx,cy 좌표 (박스 아래)
    cv2.putText(img, coord_text, (x1 + 2, y2 + 14),
                FONT, FONT_SCALE, color, THICKNESS)


def draw_hud(img: np.ndarray, det_fps: float, cap_fps: float,
             n_monsters: int, n_adena: int,
             target: Detection):
    """좌상단 HUD: FPS / 탐지수 / 타겟 좌표."""
    lines = [
        f"CAP  {cap_fps:.1f} fps",
        f"DET  {det_fps:.1f} fps",
        f"Monster: {n_monsters}  Adena: {n_adena}",
    ]
    if target:
        lines.append(f"TARGET  cx={target.cx}  cy={target.cy}"
                     f"  conf={target.confidence:.2f}")
    else:
        lines.append("TARGET  none")

    y = 20
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(img, (8, y - th - 2), (8 + tw + 4, y + 2),
                      COLOR_BG, -1)
        cv2.putText(img, line, (10, y),
                    FONT, FONT_SCALE, COLOR_TEXT, THICKNESS)
        y += th + 8


def main():
    # ── config 로드 ───────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # 모니터 인수 처리
    args = sys.argv[1:]
    mon_idx = cfg["capture"]["monitor"]
    if "--monitor" in args:
        idx = args.index("--monitor")
        if idx + 1 < len(args):
            mon_idx = int(args[idx + 1])

    # ── 캡처 초기화 ───────────────────────────────────────────
    cap = ScreenCapture(monitor=mon_idx)
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

    # ── ROI ───────────────────────────────────────────────────
    rcfg = cfg["roi"]
    roi = None
    if rcfg.get("enabled", False) and rcfg.get("width", 0) > 0:
        roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])
        print(f"[ROI] 활성: {roi}")
    else:
        print("[ROI] 비활성 (전체 화면 탐지)")

    # ── 시각화 창 ─────────────────────────────────────────────
    WIN = "YOLO Detect Test  [Q/ESC=종료]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)   # 표시 창 크기 (원본 절반)

    print()
    print("[Start] 탐지 시작. Q 또는 ESC 로 종료.")
    print("        바운딩박스가 실제 몬스터 위에 정확히 표시되는지 확인하세요.")
    print()

    # ── 로그용 이전 탐지 결과 ─────────────────────────────────
    prev_log_t  = 0.0
    LOG_INTERVAL = 0.5   # 0.5초마다 로그 출력

    try:
        while True:
            # ── 캡처 ──────────────────────────────────────────
            frame = cap.capture()
            if frame is None:
                continue

            # ── YOLO 탐지 ─────────────────────────────────────
            detections = det.detect(frame)

            # ROI 필터
            if roi:
                rx, ry, rw, rh = roi
                detections = [
                    d for d in detections
                    if rx <= d.cx <= rx + rw and ry <= d.cy <= ry + rh
                ]

            monsters = [d for d in detections if d.class_id == 0]
            adenas   = [d for d in detections if d.class_id == 1]

            # ── 타겟 선택 (confidence 최고) ───────────────────
            target = select_by_confidence(detections)

            # ── 시각화 ────────────────────────────────────────
            vis = frame.copy()

            # ROI 표시
            if roi:
                rx, ry, rw, rh = roi
                cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh),
                               (255, 200, 0), 1)

            # 아데나
            for d in adenas:
                draw_detection(vis, d, COLOR_ADENA,
                               f"adena {d.confidence:.2f}")

            # 몬스터 (타겟 아닌 것)
            for d in monsters:
                if target and d.cx == target.cx and d.cy == target.cy:
                    continue
                draw_detection(vis, d, COLOR_MONSTER,
                               f"monster {d.confidence:.2f}")

            # 타겟 (초록, 더 두껍게)
            if target:
                x1, y1, x2, y2 = target.tlbr
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_TARGET, 3)
                cv2.drawMarker(vis, (target.cx, target.cy),
                               COLOR_TARGET, cv2.MARKER_CROSS, 16, 2)
                label = f"TARGET {target.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, FONT, FONT_SCALE, THICKNESS)
                cv2.rectangle(vis,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              COLOR_TARGET, -1)
                cv2.putText(vis, label, (x1 + 2, y1 - 4),
                            FONT, FONT_SCALE, COLOR_BG, THICKNESS)
                cv2.putText(vis, f"cx={target.cx} cy={target.cy}",
                            (x1 + 2, y2 + 14),
                            FONT, FONT_SCALE, COLOR_TARGET, THICKNESS)

            # HUD
            draw_hud(vis, det.fps, cap.fps,
                     len(monsters), len(adenas), target)

            # 창 표시
            cv2.imshow(WIN, vis)

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
                            d.cx == target.cx and
                            d.cy == target.cy
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
                    print(f"  탐지 없음  DET {det.fps}fps")

            # ── 키 입력 ───────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q or ESC
                break

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        cv2.destroyAllWindows()
        print("[Stop] 종료 완료")


if __name__ == "__main__":
    main()
