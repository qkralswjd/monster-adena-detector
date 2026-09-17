"""
detection_validator.py
======================
실전 필드 YOLO 탐지 성능 검증 전용 도구.
코드 수정 없음 — config.json 설정 그대로 사용.

목적:
  사용자가 게임 필드에서 직접 이동하는 동안
  YOLO가 살아있는 몬스터를 얼마나 정확·안정적으로 탐지하는지 검증.

기록 항목 (프레임마다):
  timestamp / bbox / cx / cy / width / height / confidence

검증 항목:
  - 위치별 탐지 (중앙/좌/우/상/하/경계)
  - 크기별 탐지 (소형/중형/대형)
  - 연속 탐지 안정성 (깜빡임 추적)
  - confidence 분포
  - 이동 중 탐지 지속성

사용법:
  python detection_validator.py           # 기본: 실시간 탐지 + 자동 저장
  python detection_validator.py --no-gui  # GUI 없이 로그만 출력
  python detection_validator.py --conf 0.15  # confidence threshold 오버라이드

종료: Q / ESC / Ctrl+C → 세션 자동 저장

수정 금지: main.py, state_machine.py, controller.py, target_selector.py, detector.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

# ── 경로 설정 ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection


# ══════════════════════════════════════════════════════════════════════════════
# 설정 상수
# ══════════════════════════════════════════════════════════════════════════════

ROI_X, ROI_Y = 372, 259
ROI_W, ROI_H = 1135, 472

# 위치 영역 정의 (ROI 기준 픽셀)
ZONE_LEFT_EDGE  = 120   # cx < 120  → LEFT_EDGE
ZONE_RIGHT_EDGE = 1015  # cx > 1015 → RIGHT_EDGE
ZONE_BTM_EDGE   = 380   # cy > 380  → BTM_EDGE
ZONE_TOP_EDGE   = 80    # cy < 80   → TOP_EDGE

# 크기 기준 (bbox 픽셀)
SIZE_SMALL  = 40   # w*h < 40*40   → SMALL
SIZE_LARGE  = 100  # w or h > 100  → LARGE

# 연속 탐지 추적 파라미터
TRACK_DIST_THRESH = 120   # 같은 몬스터로 판단하는 최대 픽셀 거리
TRACK_MISS_FRAMES = 8     # 미탐 프레임이 이 수 이상이면 새 트랙으로 간주


# ══════════════════════════════════════════════════════════════════════════════
# 미탐 원인 분류
# ══════════════════════════════════════════════════════════════════════════════

MISS_TYPES = {
    'A': 'ROI 밖',
    'B': 'ROI 경계 clipping',
    'C': '화면 가장자리 clipping',
    'D': '배경 동화 (풀숲/흙/바위/그림자)',
    'E': '몬스터 크기 너무 작음',
    'F': '몬스터 이동/자세 변화',
    'G': '부분 가림',
    'H': 'confidence 불안정 (0.15~0.25 구간)',
    'I': '기타',
}


# ══════════════════════════════════════════════════════════════════════════════
# 트랙 (연속 탐지 추적)
# ══════════════════════════════════════════════════════════════════════════════

class Track:
    """단일 몬스터의 연속 탐지 이력."""

    _id_counter = 0

    def __init__(self, det: Detection, frame_no: int):
        Track._id_counter += 1
        self.track_id    = Track._id_counter
        self.cx          = det.cx
        self.cy          = det.cy

        self.det_frames:  List[int]   = [frame_no]   # 탐지된 프레임 번호
        self.miss_frames: List[int]   = []            # 미탐 프레임 번호
        self.confs:       List[float] = [det.confidence]
        self.last_det_frame = frame_no
        self.last_cx     = det.cx
        self.last_cy     = det.cy
        self.alive       = True   # 아직 추적 중

    def update_det(self, det: Detection, frame_no: int):
        self.last_det_frame = frame_no
        self.last_cx = det.cx
        self.last_cy = det.cy
        self.det_frames.append(frame_no)
        self.confs.append(det.confidence)

    def update_miss(self, frame_no: int):
        self.miss_frames.append(frame_no)

    def consecutive_miss(self, frame_no: int) -> int:
        """현재 프레임까지 연속 미탐 수."""
        return frame_no - self.last_det_frame

    def stats(self) -> Dict[str, Any]:
        total = len(self.det_frames) + len(self.miss_frames)
        recall = len(self.det_frames) / total if total else 0
        # 연속 미탐 최대 구간
        all_frames = sorted(self.det_frames + self.miss_frames)
        max_gap = 0
        if self.miss_frames:
            # 미탐 구간을 연속 구간으로 묶어 최대 길이 계산
            cur_gap = 0
            prev_type = 'det'  # det or miss
            for f in all_frames:
                if f in self.miss_frames:
                    cur_gap += 1
                    max_gap = max(max_gap, cur_gap)
                else:
                    cur_gap = 0
        return {
            'track_id':   self.track_id,
            'det_count':  len(self.det_frames),
            'miss_count': len(self.miss_frames),
            'recall':     round(recall, 3),
            'avg_conf':   round(float(np.mean(self.confs)), 3) if self.confs else 0,
            'min_conf':   round(float(np.min(self.confs)), 3) if self.confs else 0,
            'max_conf':   round(float(np.max(self.confs)), 3) if self.confs else 0,
            'max_gap':    max_gap,
            'start_cx':   self.cx,
            'start_cy':   self.cy,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 위치·크기 분류 유틸
# ══════════════════════════════════════════════════════════════════════════════

def classify_zone(cx: int, cy: int) -> str:
    """ROI 좌표 기준으로 위치 영역 분류."""
    tags = []
    if cx < ZONE_LEFT_EDGE:   tags.append('LEFT_EDGE')
    elif cx > ZONE_RIGHT_EDGE: tags.append('RIGHT_EDGE')
    else:                      tags.append('CENTER_H')
    if cy < ZONE_TOP_EDGE:     tags.append('TOP_EDGE')
    elif cy > ZONE_BTM_EDGE:   tags.append('BTM_EDGE')
    else:                      tags.append('MID_V')
    return ','.join(tags)


def classify_size(w: int, h: int) -> str:
    """bbox 픽셀 크기 분류."""
    area = w * h
    if area < SIZE_SMALL * SIZE_SMALL:
        return 'SMALL'
    elif w > SIZE_LARGE or h > SIZE_LARGE:
        return 'LARGE'
    else:
        return 'MID'


def edge_clipping(cx: int, cy: int, w: int, h: int) -> str:
    """ROI 경계 clipping 여부 판단."""
    clips = []
    if cx - w // 2 < 5:          clips.append('LEFT')
    if cx + w // 2 > ROI_W - 5:  clips.append('RIGHT')
    if cy - h // 2 < 5:          clips.append('TOP')
    if cy + h // 2 > ROI_H - 5:  clips.append('BTM')
    return ','.join(clips) if clips else ''


# ══════════════════════════════════════════════════════════════════════════════
# 세션 기록기
# ══════════════════════════════════════════════════════════════════════════════

class ValidationSession:
    """탐지 검증 세션 — 프레임별 기록 + 통계 생성."""

    def __init__(self, session_dir: str, conf_threshold: float):
        self.session_dir    = session_dir
        self.conf_threshold = conf_threshold
        os.makedirs(session_dir, exist_ok=True)

        self.frame_log: List[Dict]  = []   # 프레임별 원시 기록
        self.tracks:    List[Track] = []   # 완료된 트랙
        self.active_tracks: List[Track] = []  # 현재 추적 중

        self.frame_no  = 0
        self.start_time = time.time()

        # VideoWriter (화면 녹화)
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_lock = threading.Lock()

        # 실시간 통계
        self._det_count   = 0
        self._miss_count  = 0

    def init_video(self, w: int, h: int, fps: float = 10.0):
        """녹화용 VideoWriter 초기화."""
        path = os.path.join(self.session_dir, 'screen.mp4')
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        print(f"[REC] 녹화 시작: {path}  ({w}x{h} @ {fps:.0f}fps)")

    def write_frame(self, frame: np.ndarray):
        if self._writer and self._writer.isOpened():
            with self._writer_lock:
                self._writer.write(frame)

    def record_frame(self, dets: List[Detection], ts: float):
        """프레임 결과 기록 + 트랙 업데이트."""
        self.frame_no += 1
        fn = self.frame_no

        monsters = [d for d in dets if d.class_id == 0]

        # ── 프레임 로그 ──────────────────────────────────────────────
        frame_entry = {
            'frame':     fn,
            'ts':        round(ts - self.start_time, 3),
            'n_monster': len(monsters),
            'dets': []
        }
        for d in monsters:
            zone  = classify_zone(d.cx, d.cy)
            size  = classify_size(d.w, d.h)
            clip  = edge_clipping(d.cx, d.cy, d.w, d.h)
            frame_entry['dets'].append({
                'cx': d.cx, 'cy': d.cy,
                'w':  d.w,  'h':  d.h,
                'conf': d.confidence,
                'zone': zone, 'size': size, 'clip': clip,
            })
        self.frame_log.append(frame_entry)

        # ── 트랙 업데이트 ────────────────────────────────────────────
        matched_track_ids = set()
        matched_det_indices = set()

        # 각 활성 트랙 → 가장 가까운 탐지와 매칭
        for track in self.active_tracks:
            best_dist = TRACK_DIST_THRESH
            best_di   = None
            for di, d in enumerate(monsters):
                if di in matched_det_indices:
                    continue
                dist = ((d.cx - track.last_cx) ** 2 +
                        (d.cy - track.last_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_di   = di
            if best_di is not None:
                track.update_det(monsters[best_di], fn)
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(best_di)
                self._det_count += 1
            else:
                track.update_miss(fn)
                self._miss_count += 1
                # 연속 미탐이 TRACK_MISS_FRAMES 초과 → 트랙 종료
                if track.consecutive_miss(fn) >= TRACK_MISS_FRAMES:
                    track.alive = False

        # 종료된 트랙 정리
        dead = [t for t in self.active_tracks if not t.alive]
        self.tracks.extend(dead)
        self.active_tracks = [t for t in self.active_tracks if t.alive]

        # 매칭 안 된 탐지 → 새 트랙 생성
        for di, d in enumerate(monsters):
            if di not in matched_det_indices:
                self.active_tracks.append(Track(d, fn))
                self._det_count += 1

    def finalize(self):
        """세션 종료 — 모든 활성 트랙 수집."""
        self.tracks.extend(self.active_tracks)
        self.active_tracks = []
        if self._writer:
            self._writer.release()
            print("[REC] 녹화 저장 완료")

    def save(self):
        """세션 데이터 JSON 저장."""
        # frame_log.json
        log_path = os.path.join(self.session_dir, 'frame_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_log, f, ensure_ascii=False, indent=2)

        # tracks.json
        tracks_path = os.path.join(self.session_dir, 'tracks.json')
        with open(tracks_path, 'w', encoding='utf-8') as f:
            json.dump([t.stats() for t in self.tracks], f,
                      ensure_ascii=False, indent=2)

        print(f"[SAVE] frame_log: {log_path}")
        print(f"[SAVE] tracks:    {tracks_path}")
        return log_path, tracks_path

    @property
    def realtime_recall(self) -> float:
        total = self._det_count + self._miss_count
        return self._det_count / total if total else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FS          = 0.48
THICK       = 1

C_MONSTER   = (0,  80, 255)   # 빨강
C_TARGET    = (0, 220,   0)   # 초록
C_EDGE      = (0, 200, 255)   # 노란 오렌지 — 경계 탐지
C_BTM       = (255, 80,   0)  # 파랑 — 하단
C_TEXT      = (255, 255, 255)
C_BG        = (0, 0, 0)
C_ROI       = (200, 200, 50)


def pick_color(cx: int, cy: int) -> tuple:
    """위치에 따라 색 결정 (시각적 위치 확인 용이)."""
    if cx < ZONE_LEFT_EDGE or cx > ZONE_RIGHT_EDGE:
        return C_EDGE
    if cy > ZONE_BTM_EDGE:
        return C_BTM
    return C_MONSTER


def draw_det(img: np.ndarray, d: Detection):
    x1, y1, x2, y2 = d.tlbr
    color = pick_color(d.cx, d.cy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.drawMarker(img, (d.cx, d.cy), color, cv2.MARKER_CROSS, 10, 1)
    # conf 레이블
    label = f"{d.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, FS, THICK)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 3), FONT, FS, (0, 0, 0), THICK)
    # cx,cy
    cv2.putText(img, f"({d.cx},{d.cy})", (x1, y2 + 13),
                FONT, FS - 0.05, color, THICK)


def draw_hud(img: np.ndarray, frame_no: int, det_fps: float,
             n_det: int, recall: float, conf_thresh: float,
             session_dir: str):
    lines = [
        f"Frame:{frame_no}  DET_FPS:{det_fps:.1f}",
        f"Monster:{n_det}  conf>={conf_thresh:.2f}",
        f"Track recall: {recall*100:.1f}%",
        f"Session: {os.path.basename(session_dir)}",
        f"[Q/ESC] 종료  [S] 스냅샷",
    ]
    y = 18
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, FS, THICK)
        cv2.rectangle(img, (6, y - th - 2), (10 + tw, y + 2), C_BG, -1)
        cv2.putText(img, line, (8, y), FONT, FS, C_TEXT, THICK)
        y += th + 6


def draw_zones(img: np.ndarray):
    """ROI 내 위치 영역 경계선 표시."""
    h, w = img.shape[:2]
    # 좌측 경계
    cv2.line(img, (ZONE_LEFT_EDGE, 0), (ZONE_LEFT_EDGE, h),
             (80, 80, 80), 1, cv2.LINE_AA)
    # 우측 경계
    cv2.line(img, (ZONE_RIGHT_EDGE, 0), (ZONE_RIGHT_EDGE, h),
             (80, 80, 80), 1, cv2.LINE_AA)
    # 하단 경계
    cv2.line(img, (0, ZONE_BTM_EDGE), (w, ZONE_BTM_EDGE),
             (80, 80, 80), 1, cv2.LINE_AA)
    # 상단 경계
    cv2.line(img, (0, ZONE_TOP_EDGE), (w, ZONE_TOP_EDGE),
             (80, 80, 80), 1, cv2.LINE_AA)
    # 영역 라벨
    cv2.putText(img, 'LEFT_EDGE', (2, ZONE_BTM_EDGE - 5),
                FONT, 0.38, (80, 80, 80), 1)
    cv2.putText(img, 'RIGHT_EDGE', (ZONE_RIGHT_EDGE + 2, ZONE_BTM_EDGE - 5),
                FONT, 0.38, (80, 80, 80), 1)
    cv2.putText(img, 'BTM_EDGE', (ZONE_LEFT_EDGE + 5, ZONE_BTM_EDGE + 12),
                FONT, 0.38, (80, 150, 255), 1)


def draw_conf_bar(img: np.ndarray, confs: List[float]):
    """우측 하단에 최근 60프레임 conf 막대그래프."""
    if not confs:
        return
    bh, bw = 60, 120
    bx, by = img.shape[1] - bw - 8, img.shape[0] - bh - 8
    cv2.rectangle(img, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2),
                  (50, 50, 50), -1)
    n = len(confs)
    step = bw / max(n, 1)
    for i, c in enumerate(confs):
        bar_h = int(c * bh)
        bar_x = int(bx + i * step)
        color = (0, 220, 0) if c >= 0.20 else (
                (0, 200, 255) if c >= 0.15 else (0, 80, 200))
        cv2.rectangle(img,
                      (bar_x, by + bh - bar_h),
                      (bar_x + max(1, int(step) - 1), by + bh),
                      color, -1)
    # 0.20 기준선
    y20 = by + bh - int(0.20 * bh)
    cv2.line(img, (bx, y20), (bx + bw, y20), (0, 200, 0), 1)
    cv2.putText(img, '0.20', (bx - 30, y20 + 4), FONT, 0.35, (0, 200, 0), 1)
    cv2.putText(img, 'conf', (bx, by - 4), FONT, 0.35, (180, 180, 180), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 스냅샷 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_snapshot(img: np.ndarray, dets: List[Detection],
                  session_dir: str, frame_no: int):
    snap_dir = os.path.join(session_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    ts = datetime.now().strftime('%H%M%S_%f')[:10]
    path = os.path.join(snap_dir, f'snap_{ts}_f{frame_no}.jpg')
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    meta = {
        'frame_no': frame_no,
        'timestamp': ts,
        'detections': [
            {'cx': d.cx, 'cy': d.cy, 'w': d.w, 'h': d.h,
             'conf': d.confidence, 'zone': classify_zone(d.cx, d.cy)}
            for d in dets if d.class_id == 0
        ]
    }
    with open(path.replace('.jpg', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[SNAP] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 로그 출력
# ══════════════════════════════════════════════════════════════════════════════

class DetLogger:
    def __init__(self, log_path: str):
        self._path = log_path
        self._f    = open(log_path, 'w', encoding='utf-8', buffering=1)
        self._lock = threading.Lock()
        print(f"[LOG] 탐지 로그: {log_path}")

    def log(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:12]
        line = f"[{ts}] {msg}"
        with self._lock:
            print(line)
            self._f.write(line + '\n')

    def close(self):
        self._f.flush()
        self._f.close()


# ══════════════════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='YOLO 탐지 검증 도구')
    parser.add_argument('--no-gui',  action='store_true', help='GUI 없이 로그만')
    parser.add_argument('--conf',    type=float, default=None,
                        help='confidence threshold 오버라이드 (기본: config.json)')
    parser.add_argument('--monitor', type=int, default=None,
                        help='캡처 모니터 인덱스 오버라이드')
    parser.add_argument('--imgsz',   type=int, default=None,
                        help='YOLO imgsz 오버라이드 (기본: config.json)')
    args = parser.parse_args()

    # ── config 로드 ──────────────────────────────────────────────────
    cfg_path = os.path.join(_DIR, 'config.json')
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)

    dcfg = cfg['detector']
    rcfg = cfg['roi']
    tcfg = cfg['target']

    conf_threshold = args.conf if args.conf is not None else dcfg['confidence']
    mon_idx        = args.monitor if args.monitor is not None else cfg['capture']['monitor']
    img_size       = args.imgsz  if args.imgsz  is not None else dcfg['img_size']
    min_conf_track = tcfg.get('min_conf', 0.20)

    print("=" * 60)
    print("  YOLO 탐지 검증 도구")
    print("=" * 60)
    print(f"  모델:       {dcfg['model']}")
    print(f"  conf:       {conf_threshold}  (Tracker min_conf: {min_conf_track})")
    print(f"  imgsz:      {img_size}")
    print(f"  IoU:        {dcfg['iou_threshold']}")
    print(f"  device:     {dcfg['device']}")
    print(f"  ROI:        x={rcfg['x']} y={rcfg['y']} "
          f"w={rcfg['width']} h={rcfg['height']}")
    print("=" * 60)
    print()

    # ── 세션 디렉토리 생성 ───────────────────────────────────────────
    ts_str   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    sess_dir = os.path.join(_DIR, 'detection_sessions', ts_str)
    os.makedirs(sess_dir, exist_ok=True)
    print(f"[SESSION] {sess_dir}")

    session = ValidationSession(sess_dir, conf_threshold)
    logger  = DetLogger(os.path.join(sess_dir, 'det_log.txt'))

    # ── ROI 설정 ─────────────────────────────────────────────────────
    roi_cfg = None
    if rcfg.get('enabled', False):
        roi_cfg = {
            'enabled': True,
            'x': rcfg['x'], 'y': rcfg['y'],
            'width': rcfg['width'], 'height': rcfg['height'],
        }

    # ── 캡처 초기화 ─────────────────────────────────────────────────
    cap = ScreenCapture(monitor=mon_idx, roi=roi_cfg)
    time.sleep(0.3)

    # ── YOLO 초기화 ─────────────────────────────────────────────────
    model_path = dcfg['model']
    if not os.path.isabs(model_path):
        model_path = os.path.join(_DIR, model_path)

    if not os.path.exists(model_path):
        # v2 없으면 v1-2로 자동 대체
        fallback = os.path.join(_DIR, 'runs/detect/monster_v1-2/weights/best.pt')
        print(f"[WARN] 모델 없음: {model_path}")
        print(f"[WARN] 대체 모델 사용: {fallback}")
        model_path = fallback

    det = YOLODetector(
        model_path    = model_path,
        confidence    = conf_threshold,
        iou_threshold = dcfg['iou_threshold'],
        device        = dcfg['device'],
        img_size      = img_size,
    )

    # ── VideoWriter (ROI 크기로) ─────────────────────────────────────
    session.init_video(cap.width, cap.height, fps=10.0)

    # ── GUI 창 ───────────────────────────────────────────────────────
    WIN = 'YOLO 탐지 검증  [Q/ESC=종료  S=스냅샷]'
    show_gui = not args.no_gui
    if show_gui:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(cap.width, 1280), min(cap.height, 530))

    # ── 최근 conf 이력 (그래프용) ────────────────────────────────────
    recent_confs: deque = deque(maxlen=60)

    # ── 로그 주기 ────────────────────────────────────────────────────
    LOG_INTERVAL  = 1.0   # 초 (정기 로그)
    MISS_LOG_GAP  = 3     # 연속 미탐이 이 프레임 이상이면 즉시 로그
    prev_log_t    = 0.0
    frame_no      = 0
    _prev_was_det = False  # 직전 프레임 탐지 여부 (즉시 로그용)

    # ── 안내 출력 ────────────────────────────────────────────────────
    print()
    print("  ▶ 게임 필드에서 캐릭터를 자유롭게 이동하세요.")
    print("  ▶ 몬스터 주변을 다양하게 접근 — 중앙/좌/우/하단/경계")
    print("  ▶ Q 또는 ESC 로 종료 → 자동 저장")
    print("  ▶ S 로 현재 프레임 스냅샷 저장")
    print()
    print("  [색상 가이드]")
    print("   빨강  = 중앙 몬스터")
    print("   오렌지 = 좌/우 경계 몬스터")
    print("   파랑  = 하단 경계 몬스터")
    print("  [회색 점선] = 위치 영역 경계")
    print()

    try:
        while True:
            t_start = time.time()

            # ── 캡처 ────────────────────────────────────────────────
            frame, is_new = cap.capture()
            if frame is None:
                time.sleep(0.005)
                continue
            if not is_new:
                time.sleep(0.002)
                continue

            frame_no += 1

            # ── YOLO 탐지 ────────────────────────────────────────────
            dets = det.detect(frame)
            monsters = [d for d in dets if d.class_id == 0]

            # ── 세션 기록 ────────────────────────────────────────────
            session.record_frame(dets, t_start)

            # ── conf 이력 ────────────────────────────────────────────
            for d in monsters:
                recent_confs.append(d.confidence)

            # ── 화면 녹화 ────────────────────────────────────────────
            session.write_frame(frame)

            # ── 즉시 로그: 탐지→미탐 전환 시 ───────────────────────
            if _prev_was_det and not monsters:
                logger.log(f"  ↳ [MISS 시작] frame={frame_no} "
                           f"recall:{session.realtime_recall*100:.0f}%")
            elif not _prev_was_det and monsters:
                logger.log(f"  ↳ [DET 재개]  frame={frame_no} "
                           f"conf={monsters[0].confidence:.3f} "
                           f"cx={monsters[0].cx} cy={monsters[0].cy}")
            _prev_was_det = bool(monsters)

            # ── 로그 (1초마다) ───────────────────────────────────────
            now = time.time()
            if now - prev_log_t >= LOG_INTERVAL:
                prev_log_t = now
                if monsters:
                    parts = []
                    for d in monsters:
                        zone = classify_zone(d.cx, d.cy)
                        size = classify_size(d.w, d.h)
                        clip = edge_clipping(d.cx, d.cy, d.w, d.h)
                        clip_str = f" CLIP:{clip}" if clip else ""
                        parts.append(
                            f"cx={d.cx:4d} cy={d.cy:3d} "
                            f"w={d.w:3d} h={d.h:3d} "
                            f"conf={d.confidence:.3f} "
                            f"[{zone}][{size}]{clip_str}"
                        )
                    logger.log(
                        f"DET×{len(monsters)} | " + " | ".join(parts) +
                        f" | FPS:{det.fps:.1f} | recall:{session.realtime_recall*100:.0f}%"
                    )
                else:
                    # 미탐 원인 자동 추론 (직전 트랙 위치 기반)
                    miss_hint = ""
                    if session.active_tracks:
                        worst = max(session.active_tracks,
                                    key=lambda t: t.consecutive_miss(frame_no))
                        gap = worst.consecutive_miss(frame_no)
                        cx, cy = worst.last_cx, worst.last_cy
                        # 간단 원인 추론
                        if cx < ZONE_LEFT_EDGE or cx > ZONE_RIGHT_EDGE:
                            hint = "B?경계"
                        elif cy > ZONE_BTM_EDGE:
                            hint = "D?하단배경"
                        elif gap >= 5:
                            hint = "F?자세변화/이동"
                        else:
                            hint = "H?conf불안정"
                        miss_hint = f" | MISS×{gap}프레임 트랙@({cx},{cy}) [{hint}]"
                    logger.log(
                        f"MISS    FPS:{det.fps:.1f} "
                        f"recall:{session.realtime_recall*100:.0f}%"
                        f"{miss_hint}"
                    )

            # ── GUI ──────────────────────────────────────────────────
            if show_gui:
                vis = frame.copy()

                # 영역 경계선
                draw_zones(vis)

                # 탐지 박스
                for d in monsters:
                    draw_det(vis, d)

                # HUD
                draw_hud(vis, frame_no, det.fps,
                         len(monsters), session.realtime_recall,
                         conf_threshold, sess_dir)

                # conf 막대그래프
                draw_conf_bar(vis, list(recent_confs))

                # 연속 미탐 경고
                for track in session.active_tracks:
                    gap = track.consecutive_miss(frame_no)
                    if gap >= 3:
                        cx, cy = track.last_cx, track.last_cy
                        cv2.putText(
                            vis,
                            f"MISS×{gap}",
                            (max(0, cx - 30), max(20, cy - 10)),
                            FONT, 0.55, (0, 0, 255), 2
                        )

                cv2.imshow(WIN, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\n[종료] Q/ESC 입력")
                    break
                elif key in (ord('s'), ord('S')):
                    save_snapshot(vis, dets, sess_dir, frame_no)

    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C")
    finally:
        if show_gui:
            cv2.destroyAllWindows()

        session.finalize()
        log_path, tracks_path = session.save()
        logger.close()

        # ── 즉시 요약 출력 ───────────────────────────────────────────
        print()
        print("=" * 60)
        print("  탐지 검증 세션 종료 — 요약")
        print("=" * 60)
        print(f"  총 프레임:  {session.frame_no}")
        print(f"  conf 기준:  {conf_threshold}")
        print()

        all_dets = [d for f in session.frame_log for d in f['dets']]
        if all_dets:
            confs_all = [d['conf'] for d in all_dets]
            print(f"  총 탐지:    {len(all_dets)}건")
            print(f"  avg conf:   {np.mean(confs_all):.3f}")
            print(f"  min conf:   {np.min(confs_all):.3f}")
            print(f"  conf>=0.20: {sum(1 for c in confs_all if c>=0.20)} "
                  f"({sum(1 for c in confs_all if c>=0.20)/len(confs_all)*100:.0f}%)")
            print(f"  conf>=0.15: {sum(1 for c in confs_all if c>=0.15)} "
                  f"({sum(1 for c in confs_all if c>=0.15)/len(confs_all)*100:.0f}%)")
            print()

            # 위치별
            zones = {}
            for d in all_dets:
                z = d['zone']
                zones[z] = zones.get(z, 0) + 1
            print("  [위치별 탐지 수]")
            for z, c in sorted(zones.items(), key=lambda x: -x[1]):
                print(f"    {z}: {c}")
            print()

            # 크기별
            sizes = {}
            for d in all_dets:
                s = d['size']
                sizes[s] = sizes.get(s, 0) + 1
            print("  [크기별 탐지 수]")
            for s, c in sorted(sizes.items(), key=lambda x: -x[1]):
                print(f"    {s}: {c}")
            print()

            # 경계 clipping 발생
            clips = [d for d in all_dets if d['clip']]
            print(f"  경계 clipping 발생: {len(clips)}건")
            clip_types = {}
            for d in clips:
                clip_types[d['clip']] = clip_types.get(d['clip'], 0) + 1
            for ct, c in clip_types.items():
                print(f"    {ct}: {c}건")
            print()

        # 트랙 통계
        all_tracks = session.tracks
        if all_tracks:
            recalls = [t.stats()['recall'] for t in all_tracks]
            max_gaps = [t.stats()['max_gap'] for t in all_tracks]
            print(f"  트랙 수:    {len(all_tracks)}")
            print(f"  평균 recall: {np.mean(recalls)*100:.1f}%")
            print(f"  최대 연속 미탐: {max(max_gaps)}프레임")
            unstable = [t for t in all_tracks if t.stats()['recall'] < 0.7]
            print(f"  불안정 트랙(recall<70%): {len(unstable)}")
            print()

        print(f"  저장 위치: {sess_dir}")
        print()
        print("  → 세션 종료 후 분석:")
        print(f"    python analyze_detection_session.py {sess_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()
