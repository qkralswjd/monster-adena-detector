"""
visualizer.py
-------------
화면에 탐지 결과 / 타겟 / HUD를 그리는 모듈.
"""

import cv2
import numpy as np
from typing import List, Optional
from detector import Detection


# 색상 (BGR)
CLR_MONSTER = (0, 200, 255)   # 주황 - 일반 몬스터
CLR_TARGET  = (0, 255, 0)     # 초록 - 현재 타겟
CLR_DEAD    = (0, 0, 255)     # 빨강 - 소실/사망 중
CLR_HUD     = (255, 255, 255) # 흰색 - HUD 텍스트
CLR_SHADOW  = (0, 0, 0)       # 검정 - 텍스트 그림자


def _text(img, txt, x, y, color=CLR_HUD, scale=0.55, thick=1):
    """그림자 있는 텍스트."""
    cv2.putText(img, txt, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, CLR_SHADOW, thick + 1, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw(frame: np.ndarray,
         detections: List[Detection],
         target: Optional[Detection],
         miss_elapsed: float = 0.0,
         detector_fps: float = 0.0,
         capture_fps: float = 0.0) -> np.ndarray:
    """
    탐지 박스 + 타겟 박스 + HUD 그리기.

    Args:
        frame        : 원본 프레임 (BGR)
        detections   : 이번 프레임 탐지 목록
        target       : 현재 타겟 (None이면 미지정)
        miss_elapsed : 타겟 소실 경과 시간 (0이면 정상)
        detector_fps : 탐지기 FPS
        capture_fps  : 캡처 FPS
    Returns:
        그려진 프레임
    """
    out = frame.copy()

    # ── 일반 몬스터 박스 ──────────────────────────────────────
    for d in detections:
        is_target = (target is not None and
                     d.x == target.x and d.y == target.y)
        if is_target:
            continue  # 타겟은 아래서 따로 그림
        x, y, w, h = d.x, d.y, d.w, d.h
        cv2.rectangle(out, (x, y), (x+w, y+h), CLR_MONSTER, 1)
        _text(out, f"{d.confidence:.2f}", x, y - 5,
              color=CLR_MONSTER, scale=0.45)

    # ── 타겟 박스 ─────────────────────────────────────────────
    if target is not None:
        x, y, w, h = target.x, target.y, target.w, target.h

        if miss_elapsed > 0:
            # 소실 중 → 빨간 점선 효과 (두께 줄이기로 대체)
            color = CLR_DEAD
            thick = 1
            label = f"MISSING {miss_elapsed:.1f}s"
        else:
            color = CLR_TARGET
            thick = 2
            label = f"TARGET  {target.confidence:.2f}"

        cv2.rectangle(out, (x, y), (x+w, y+h), color, thick)

        # 코너 강조
        corner = 12
        cv2.line(out, (x, y), (x+corner, y), color, 3)
        cv2.line(out, (x, y), (x, y+corner), color, 3)
        cv2.line(out, (x+w, y), (x+w-corner, y), color, 3)
        cv2.line(out, (x+w, y), (x+w, y+corner), color, 3)
        cv2.line(out, (x, y+h), (x+corner, y+h), color, 3)
        cv2.line(out, (x, y+h), (x, y+h-corner), color, 3)
        cv2.line(out, (x+w, y+h), (x+w-corner, y+h), color, 3)
        cv2.line(out, (x+w, y+h), (x+w, y+h-corner), color, 3)

        # 중심점
        cv2.circle(out, (target.cx, target.cy), 4, color, -1)

        # 라벨
        _text(out, label, x, y - 8, color=color, scale=0.5, thick=1)

    # ── HUD (좌상단) ──────────────────────────────────────────
    hud_lines = [
        f"Monsters : {len(detections)}",
        f"Det FPS  : {detector_fps:.1f}",
        f"Cap FPS  : {capture_fps:.1f}",
        f"Target   : {'YES' if target else 'NONE'}",
    ]
    for i, line in enumerate(hud_lines):
        _text(out, line, 10, 20 + i * 20, scale=0.5)

    # ── 조작 안내 (우하단) ────────────────────────────────────
    h, w = out.shape[:2]
    tips = ["Click: 타겟 지정", "C: 해제", "R: ROI", "Q/ESC: 종료"]
    for i, t in enumerate(reversed(tips)):
        _text(out, t, w - 160, h - 10 - i * 18, scale=0.42)

    return out
