"""
visualizer.py
-------------
추적 결과 시각화 모듈.

표시 항목:
  - 모든 트랙 박스 (초록)
  - 타겟 박스 (빨강, 굵게)
  - 타겟 이동 경로 (선)
  - 예측 위치 (점선 박스)
  - HUD (FPS, 상태, confidence, 속도 등)
  - 타겟 스냅샷 (우하단)
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from byte_tracker import Track
from target_tracker import TargetInfo, TARGET_TRACKING, TARGET_LOST, TARGET_CONFIRMED_LOST

# 색상
C_TRACK    = (50, 220, 50)      # 초록 - 일반 트랙
C_TARGET   = (0, 60, 255)       # 빨강 - 타겟
C_LOST     = (0, 165, 255)      # 주황 - lost
C_PRED     = (255, 200, 0)      # 하늘 - 예측 위치
C_PATH     = (0, 255, 220)      # 노랑 - 경로
C_HUD      = (200, 255, 200)    # 연두 - HUD 텍스트
C_VELOCITY = (255, 100, 255)    # 보라 - 속도 화살표

FONT       = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    def __init__(self):
        self._path_points: List[Tuple[int,int]] = []

    def draw(self,
             frame: np.ndarray,
             tracks: List[Track],
             target_info: Optional[TargetInfo],
             path_points: List[Tuple[int,int]],
             snapshot: Optional[np.ndarray],
             capture_fps: float,
             detect_fps: float) -> np.ndarray:

        out = frame.copy()

        # 1. 일반 트랙 박스
        for t in tracks:
            if target_info and t.track_id == target_info.track_id:
                continue  # 타겟은 따로 그림
            self._draw_track(out, t)

        # 2. 이동 경로
        self._draw_path(out, path_points)

        # 3. 타겟 박스 + 정보
        if target_info:
            self._draw_target(out, target_info)

        # 4. HUD
        self._draw_hud(out, tracks, target_info, capture_fps, detect_fps)

        # 5. 타겟 스냅샷
        if snapshot is not None:
            self._draw_snapshot(out, snapshot)

        return out

    # ------------------------------------------------------------------

    def _draw_track(self, frame, track: Track):
        x, y, w, h = track.predicted_bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), C_TRACK, 1)
        label = f"#{track.track_id} {track.score:.2f}"
        cv2.putText(frame, label, (x, y-4),
                    FONT, 0.4, C_TRACK, 1, cv2.LINE_AA)

    def _draw_target(self, frame, info: TargetInfo):
        x = info.cx - info.w//2
        y = info.cy - info.h//2

        if info.state == TARGET_TRACKING:
            color = C_TARGET
            thick = 3
        elif info.state == TARGET_LOST:
            color = C_LOST
            thick = 2
        else:
            color = (100,100,100)
            thick = 1

        # 박스
        cv2.rectangle(frame, (x,y), (x+info.w, y+info.h), color, thick)

        # 중심점
        cv2.circle(frame, (info.cx, info.cy), 5, color, -1)

        # 속도 화살표
        if info.speed > 1:
            ex = int(info.cx + info.vx * 5)
            ey = int(info.cy + info.vy * 5)
            cv2.arrowedLine(frame, (info.cx,info.cy), (ex,ey),
                            C_VELOCITY, 2, tipLength=0.3)

        # 예측 위치 (lost일 때)
        if info.state == TARGET_LOST and info.pred_cx and info.pred_cy:
            px = info.pred_cx - info.w//2
            py = info.pred_cy - info.h//2
            cv2.rectangle(frame, (px,py), (px+info.w,py+info.h),
                          C_PRED, 1)
            cv2.putText(frame, "PRED", (px, py-4),
                        FONT, 0.4, C_PRED, 1)

        # 레이블
        label = f"TARGET #{info.track_id} | {info.state}"
        conf_label = f"conf:{info.target_confidence:.2f} spd:{info.speed:.1f}"
        self._put_bg_text(frame, label, x, y-18, color)
        self._put_bg_text(frame, conf_label, x, y-4, color, scale=0.4)

    def _draw_path(self, frame, points: List[Tuple[int,int]]):
        if len(points) < 2:
            return
        for i in range(1, len(points)):
            alpha = i / len(points)
            color = (
                int(C_PATH[0] * alpha),
                int(C_PATH[1] * alpha),
                int(C_PATH[2] * alpha)
            )
            cv2.line(frame, points[i-1], points[i], color, 2)

    def _draw_hud(self, frame, tracks, target_info, cap_fps, det_fps):
        lines = [
            f"FPS: {cap_fps:.1f}  Det: {det_fps:.1f}",
            f"Tracks: {len(tracks)}",
        ]
        if target_info:
            lines += [
                f"--- TARGET #{target_info.track_id} ---",
                f"State: {target_info.state}",
                f"Pos: ({target_info.cx}, {target_info.cy})",
                f"Speed: {target_info.speed:.1f} px/f",
                f"Vel: ({target_info.vx:.1f}, {target_info.vy:.1f})",
                f"Det conf:  {target_info.det_confidence:.2f}",
                f"Trk conf:  {target_info.track_confidence:.2f}",
                f"Tgt conf:  {target_info.target_confidence:.2f}",
            ]
        else:
            lines.append("Target: None (클릭으로 지정)")

        for i, line in enumerate(lines):
            y = 20 + i * 18
            self._put_bg_text(frame, line, 8, y, C_HUD)

        # 우상단 상태 크게
        if target_info:
            if target_info.state == TARGET_TRACKING:
                status = "TRACKING"
                color = (0, 220, 80)
            elif target_info.state == TARGET_LOST:
                status = "LOST - SEARCHING"
                color = (0, 165, 255)
            else:
                status = "TARGET LOST"
                color = (0, 0, 255)
            h, w = frame.shape[:2]
            self._put_bg_text(frame, status, w-220, 24, color, scale=0.7)

    def _draw_snapshot(self, frame, snapshot: np.ndarray):
        """우하단에 타겟 스냅샷 표시."""
        fh, fw = frame.shape[:2]
        sh, sw = snapshot.shape[:2]
        x, y = fw - sw - 10, fh - sh - 10
        frame[y:y+sh, x:x+sw] = snapshot
        cv2.rectangle(frame, (x-1,y-1), (x+sw,y+sh), (255,255,255), 1)
        cv2.putText(frame, "TARGET", (x, y-4),
                    FONT, 0.4, (255,255,255), 1)

    def _put_bg_text(self, frame, text, x, y, color,
                     scale=0.45, thickness=1):
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
        sub = frame[max(0,y-th-2):y+4, max(0,x-1):x+tw+2]
        if sub.size > 0:
            rect = np.zeros_like(sub)
            cv2.addWeighted(sub, 0.5, rect, 0.5, 0, sub)
            frame[max(0,y-th-2):y+4, max(0,x-1):x+tw+2] = sub
        cv2.putText(frame, text, (x,y), FONT, scale, color, thickness, cv2.LINE_AA)
