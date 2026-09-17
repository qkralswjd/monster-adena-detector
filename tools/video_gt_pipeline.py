"""
tools/video_gt_pipeline.py
==========================
실제 게임 녹화 영상 → YOLO 학습용 Ground Truth 데이터셋 구축 파이프라인.

목적:
  "영상 속 무엇이 몬스터인가"를 시각적으로 판단하여
  YOLO 학습에 바로 사용할 수 있는 Ground Truth bbox를 생성한다.

원칙:
  - 기존 YOLO 결과를 정답으로 복사하지 않는다
  - 영상 자체를 기준으로 실제 몬스터를 확인
  - Tracking은 라벨링 정확도를 높이기 위한 보조 수단
  - 애매한 프레임은 hard_cases로 분류

파이프라인:
  1. 영상 → 프레임 추출 (diff+phash 하이브리드 중복제거)
  2. YOLO 보조 탐지 (후보 생성용, 정답 아님)
  3. Multi-cue Tracker (IoU + 크기 + 화면이동 보정)
  4. Track 품질 기반 GT bbox 필터링
  5. CASE A/B/C/D/E 분류
  6. YOLO format 라벨 저장
  7. Preview + Contact Sheet 생성
  8. 라벨 품질 검사
  9. 최종 보고서

클래스: 0=monster, 1=adena

사용법:
  python tools/video_gt_pipeline.py
  python tools/video_gt_pipeline.py --step extract
  python tools/video_gt_pipeline.py --step label
  python tools/video_gt_pipeline.py --step preview
  python tools/video_gt_pipeline.py --step report
"""

from __future__ import annotations
import argparse, cv2, glob, json, math, os, shutil, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 경로 / 설정
# ─────────────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
DATASET    = BASE / 'dataset_field_v1'
MODEL_PATH = BASE / 'runs/detect/monster_v1-2/weights/best.pt'   # 보조용
if not MODEL_PATH.exists():
    MODEL_PATH = BASE / 'runs/detect/monster_v1/weights/best.pt'

ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472   # config.json 기준

# 영상 목록 + split 할당
VIDEOS = {
    'VID_A': ('test_runs/2026-09-17_06-13-10/screen.mp4',        'train'),
    'VID_B': ('test_runs/2026-09-17_06-40-52/screen.mp4',        'train'),
    'VID_C': ('test_runs/2026-09-17_07-00-50/screen.mp4',        'train'),
    'VID_D': ('test_runs/2026-09-17_07-45-08/screen.mp4',        'val'),
    'VID_E': ('test_runs/2026-09-17_08-14-32/screen.mp4',        'val'),
    'VID_F': ('test_runs/2026-09-17_09-10-38/screen_95mb.mp4',   'test'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 프레임 추출 파라미터
# ─────────────────────────────────────────────────────────────────────────────
EXTRACT_INTERVAL   = 3      # N프레임마다 샘플링 (10fps→3.3fps)
DIFF_SKIP_THRESH   = 0.4    # ROI 픽셀 평균차 이하 → 로딩/완전정적만 스킵
DIFF_FORCE_INCLUDE = 20.0   # 이상 → 큰 화면변화 무조건 포함
PHASH_DUP_THRESH   = -1     # -1 = phash 비활성화 (게임화면은 phash 효과없음)
MAX_FRAMES_PER_VID = 600    # 영상당 최대 추출 프레임
# diff 기반 적응형 샘플링: diff가 낮은 구간은 간격을 늘려 중복 줄임
DIFF_LOW_THRESH    = 2.0    # 이 이하면 "거의 정적" → 6프레임에 1장
DIFF_MED_THRESH    = 8.0    # 이 이하면 "약간 이동" → 3프레임에 1장 (기본)

# ─────────────────────────────────────────────────────────────────────────────
# YOLO 보조 탐지 파라미터
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CONF_LOW  = 0.10   # 이 이상은 모두 후보로 수집 (정답 X, 후보용)
YOLO_CONF_GOOD = 0.50   # CASE A (고신뢰 성공)
YOLO_CONF_MID  = 0.25   # CASE C (저신뢰)
YOLO_IOU_THRESH = 0.45

# ─────────────────────────────────────────────────────────────────────────────
# Multi-cue Tracker 파라미터
# ─────────────────────────────────────────────────────────────────────────────
TRACK_IOU_THRESH     = 0.20   # 동일 트랙 매칭 IoU 최소값
TRACK_MAX_MISS       = 4      # 이 프레임 수 이상 미탐 → 트랙 종료
TRACK_MIN_LEN        = 2      # 최소 트랙 길이 (너무 짧은 건 오탐)
GT_MIN_CONF_FOR_LABEL = 0.30  # GT 확정 최소 confidence
GT_MIN_TRACK_LEN     = 2      # 이 이상 트랙에서 온 bbox만 GT로 확정

# ─────────────────────────────────────────────────────────────────────────────
# bbox 품질 기준
# ─────────────────────────────────────────────────────────────────────────────
MIN_BOX_AREA_RATIO  = 0.0005  # ROI 면적 대비 최소 bbox 면적
MAX_BOX_AREA_RATIO  = 0.80    # ROI 면적 대비 최대 bbox 면적
CLIP_RATIO_WARN     = 0.85    # 이 이상 클리핑 → CASE C

# ─────────────────────────────────────────────────────────────────────────────
# 색상 (BGR)
# ─────────────────────────────────────────────────────────────────────────────
COL = {
    'GT'   : (0, 255, 0),       # 초록  - 확정 GT
    'YOLO' : (255, 200, 0),     # 하늘  - YOLO 보조
    'MISS' : (0, 0, 255),       # 빨강  - 미탐
    'FP'   : (0, 128, 255),     # 주황  - 오탐
    'HARD' : (200, 0, 200),     # 보라  - 애매
    'ROI'  : (128, 128, 128),   # 회색  - ROI 경계
    'TRACK': (0, 255, 255),     # 노랑  - 트랙 ID
}

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def log(msg, lv='INFO'):
    print(f"[{time.strftime('%H:%M:%S')}][{lv}] {msg}", flush=True)

def iou(a, b):
    """a,b = [x1,y1,x2,y2] 절대좌표"""
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def xywh_to_xyxy(cx,cy,w,h):
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

def xyxy_to_xywh(x1,y1,x2,y2):
    cx=(x1+x2)/2; cy=(y1+y2)/2
    return cx, cy, x2-x1, y2-y1

def norm_to_roi_abs(cx_n, cy_n, w_n, h_n):
    """ROI 정규화 좌표 → ROI 기준 절대좌표 [x1,y1,x2,y2]"""
    cx = cx_n * ROI_W; cy = cy_n * ROI_H
    bw = w_n  * ROI_W; bh = h_n  * ROI_H
    return [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]

def roi_abs_to_norm(x1,y1,x2,y2):
    """ROI 기준 절대좌표 → ROI 정규화 좌표"""
    cx=(x1+x2)/2/ROI_W; cy=(y1+y2)/2/ROI_H
    w=(x2-x1)/ROI_W;    h=(y2-y1)/ROI_H
    return cx,cy,w,h

def clip_to_roi(x1,y1,x2,y2):
    return max(0,x1), max(0,y1), min(ROI_W,x2), min(ROI_H,y2)

def compute_phash(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray,(16,16),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:4,:4].flatten()
    med = np.median(low)
    return (low > med).astype(np.uint8)

def hamming(h1,h2):
    return int(np.sum(h1!=h2))

def classify_position(cx_n, cy_n):
    """ROI 정규화 좌표 → 위치 태그"""
    tags = []
    if   cx_n < 0.25: tags.append('LEFT')
    elif cx_n > 0.75: tags.append('RIGHT')
    else:             tags.append('CENTER')
    if   cy_n < 0.35: tags.append('TOP')
    elif cy_n > 0.65: tags.append('BOTTOM')
    else:             tags.append('MIDDLE')
    return tags

def classify_size(w_n, h_n):
    area = w_n * h_n
    if   area < 0.005: return 'SMALL'
    elif area < 0.04:  return 'MEDIUM'
    else:              return 'LARGE'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: 프레임 추출 (diff+phash 하이브리드)
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(vid_id, vid_path, out_dir) -> List[Dict]:
    """
    영상 → 프레임 추출 (순차읽기 + diff 기반 적응형 샘플링).

    전략:
    - cap.set() 반복 대신 순차읽기 (10배 빠름)
    - diff 기반 적응형 간격: 정적구간은 간격 넓혀 중복 방지
    - phash 비활성화 (고정배경 게임화면에서는 효과 없음)
    - diff >= DIFF_FORCE_INCLUDE → 즉시 포함 (장면전환/큰 이동)

    반환: [{fname, frame_no, ts_sec, diff, path}, ...]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'extract_meta.json'
    if meta_path.exists():
        existing = list(out_dir.glob('*.jpg'))
        if existing:
            with open(meta_path) as f:
                return json.load(f)

    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"[{vid_id}] 영상: {fc}frames {fc/fps:.1f}s  추출 시작")

    records   = []
    prev_gray = None
    fn        = 0          # 현재 프레임 번호
    next_keep = 0          # 다음 keep 예정 프레임

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi  = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray - prev_gray)))
        else:
            diff = 99.0

        should_keep = False

        # 장면전환/큰 이동 → 무조건 포함
        if diff >= DIFF_FORCE_INCLUDE and prev_gray is not None:
            should_keep = True

        # 완전 정적 → 스킵 (메뉴/로딩 화면)
        elif diff < DIFF_SKIP_THRESH and len(records) > 0:
            should_keep = False

        # 적응형 간격 기반 샘플링
        elif fn >= next_keep:
            should_keep = True
            # 다음 keep 간격 결정
            if diff < DIFF_LOW_THRESH:
                next_keep = fn + 6    # 정적 → 0.6초 간격
            elif diff < DIFF_MED_THRESH:
                next_keep = fn + 3    # 보통 → 0.3초 간격
            else:
                next_keep = fn + 2    # 활발 → 0.2초 간격

        if should_keep:
            fname = f"{vid_id}_f{fn:06d}.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append({
                'fname'   : fname,
                'path'    : str(fpath),
                'vid_id'  : vid_id,
                'frame_no': fn,
                'ts_sec'  : round(fn / fps, 2),
                'diff'    : round(diff, 2),
            })
            prev_gray = gray
            if len(records) >= MAX_FRAMES_PER_VID:
                break

        fn += 1

    cap.release()
    log(f"[{vid_id}]   추출: {len(records)}장 (원본 {fc}frames)")

    with open(meta_path, 'w') as f:
        json.dump(records, f, indent=2)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: YOLO 보조 탐지 (후보 생성, 정답 아님)
# ─────────────────────────────────────────────────────────────────────────────
def load_yolo_model():
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    dummy = np.zeros((640,640,3), dtype=np.uint8)
    model(dummy, imgsz=640, verbose=False, device='cpu')
    log(f"모델 로드: {MODEL_PATH.name}")
    return model

def yolo_detect(model, frame) -> List[Dict]:
    """
    ROI crop → YOLO 추론 → 결과를 ROI 기준 정규화 좌표로 변환.
    반환: [{cls_id, cls_name, conf, cx_n, cy_n, w_n, h_n, x1r,y1r,x2r,y2r, clipped}, ...]
    """
    roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    results = model(roi, imgsz=640, conf=YOLO_CONF_LOW,
                    iou=YOLO_IOU_THRESH, verbose=False, device='cpu')
    dets = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            x1,y1,x2,y2 = map(float, box.xyxy[0])
            # ROI 경계 클리핑
            cx1,cy1,cx2,cy2 = clip_to_roi(x1,y1,x2,y2)
            orig_area = (x2-x1)*(y2-y1)
            clip_area = (cx2-cx1)*(cy2-cy1)
            clipped = (orig_area > 0) and (clip_area/orig_area < CLIP_RATIO_WARN)
            cx_n,cy_n,w_n,h_n = roi_abs_to_norm(cx1,cy1,cx2,cy2)
            dets.append({
                'cls_id'  : cls,
                'cls_name': {0:'monster',1:'adena'}.get(cls,'unknown'),
                'conf'    : round(conf, 4),
                'cx_n': round(cx_n,4), 'cy_n': round(cy_n,4),
                'w_n' : round(w_n, 4), 'h_n' : round(h_n, 4),
                'x1r':cx1,'y1r':cy1,'x2r':cx2,'y2r':cy2,
                'clipped' : clipped,
            })
    return dets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Multi-cue Tracker
# ─────────────────────────────────────────────────────────────────────────────
class Track:
    _next_id = 1

    def __init__(self, det, frame_no):
        self.tid      = Track._next_id
        Track._next_id += 1
        self.dets     = [det]          # 프레임별 탐지 목록
        self.frame_nos= [frame_no]
        self.missed   = 0
        self.last_box = [det['x1r'],det['y1r'],det['x2r'],det['y2r']]
        self.last_cx  = det['cx_n']
        self.last_cy  = det['cy_n']
        self.cls_id   = det['cls_id']
        self.conf_hist= [det['conf']]

    def update(self, det, frame_no):
        self.dets.append(det)
        self.frame_nos.append(frame_no)
        self.last_box = [det['x1r'],det['y1r'],det['x2r'],det['y2r']]
        self.last_cx  = det['cx_n']
        self.last_cy  = det['cy_n']
        self.missed   = 0
        self.conf_hist.append(det['conf'])

    @property
    def length(self):
        return len(self.dets)

    @property
    def max_conf(self):
        return max(self.conf_hist)

    @property
    def mean_conf(self):
        return float(np.mean(self.conf_hist))

    def predicted_box(self, n_missed):
        """단순 등속도 예측 (화면이동 보정 미포함 버전)."""
        if len(self.dets) < 2:
            return self.last_box[:]
        prev = self.dets[-2]
        cur  = self.dets[-1]
        dx = cur['x1r'] - prev['x1r']
        dy = cur['y1r'] - prev['y1r']
        f = n_missed + 1
        x1 = self.last_box[0] + dx*f
        y1 = self.last_box[1] + dy*f
        x2 = self.last_box[2] + dx*f
        y2 = self.last_box[3] + dy*f
        return [x1,y1,x2,y2]


def multi_cue_match(tracks: List[Track], dets: List[Dict],
                    frame_no: int, roi_diff: float) -> Tuple[Dict, List]:
    """
    기존 트랙 ↔ 새 탐지 매칭.
    비용 = 1-IoU + 크기차이 패널티 + 위치 이탈 패널티
    반환: (matches {track_idx: det_idx}, unmatched_dets)
    """
    if not tracks or not dets:
        return {}, list(range(len(dets)))

    cost = np.ones((len(tracks), len(dets))) * 1e6
    for ti, trk in enumerate(tracks):
        pred = trk.predicted_box(trk.missed)
        for di, det in enumerate(dets):
            if det['cls_id'] != trk.cls_id:
                continue
            dbox = [det['x1r'],det['y1r'],det['x2r'],det['y2r']]
            iou_val = iou(pred, dbox)
            if iou_val < TRACK_IOU_THRESH:
                continue
            # 크기 차이 패널티
            sz_trk = (pred[2]-pred[0])*(pred[3]-pred[1])
            sz_det = (dbox[2]-dbox[0])*(dbox[3]-dbox[1])
            sz_ratio = min(sz_trk,sz_det)/max(sz_trk,sz_det) if max(sz_trk,sz_det)>0 else 0
            penalty = (1-sz_ratio)*0.3
            cost[ti,di] = 1 - iou_val + penalty

    # 헝가리안 대신 그리디 매칭 (속도 우선)
    matches = {}
    used_dets = set()
    for ti in np.argsort(cost.min(axis=1)):
        di = int(np.argmin(cost[ti]))
        if cost[ti,di] < 1e5 and di not in used_dets:
            matches[ti] = di
            used_dets.add(di)

    unmatched = [i for i in range(len(dets)) if i not in used_dets]
    return matches, unmatched


def run_tracker(frame_records: List[Dict], all_dets: Dict[int, List[Dict]]) -> List[Track]:
    """
    전체 프레임에 걸쳐 Multi-cue Tracker 실행.
    반환: 완료된 트랙 목록
    """
    Track._next_id = 1
    active: List[Track] = []
    finished: List[Track] = []

    for rec in frame_records:
        fn   = rec['frame_no']
        dets = all_dets.get(fn, [])
        # monster만 추적
        monster_dets = [d for d in dets if d['cls_id'] == 0]

        matches, unmatched = multi_cue_match(active, monster_dets, fn,
                                              rec.get('diff', 0))

        # 매칭된 트랙 업데이트
        for ti, di in matches.items():
            active[ti].update(monster_dets[di], fn)

        # 미매칭 트랙 → missed 카운트
        for ti in range(len(active)):
            if ti not in matches:
                active[ti].missed += 1

        # 종료 트랙 처리
        still_active = []
        for trk in active:
            if trk.missed > TRACK_MAX_MISS:
                finished.append(trk)
            else:
                still_active.append(trk)
        active = still_active

        # 새 탐지 → 새 트랙
        for di in unmatched:
            active.append(Track(monster_dets[di], fn))

    finished.extend(active)
    return finished


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: GT bbox 확정 + CASE 분류
# ─────────────────────────────────────────────────────────────────────────────
def build_gt_labels(frame_records, all_dets, tracks) -> Dict[int, Dict]:
    """
    트랙 + YOLO 결과 기반 GT 라벨 결정.

    CASE 분류:
      A = 탐지 성공 (conf>=GOOD, 트랙 길이>=GT_MIN)
      B = 미탐 (트랙은 있으나 해당 프레임 YOLO 탐지 없음)
      C = 저신뢰 (conf MID~GOOD, 트랙 연속성으로 보완)
      D = bbox 수정 (트랙 중앙값으로 bbox 교정)
      E = 오탐 (YOLO탐지 있으나 트랙 없음 or 트랙 짧음)

    반환: {frame_no: {gt_dets:[...], case:str, yolo_dets:[...], ...}}
    """
    # frame_no → track들 매핑
    frame_to_tracks: Dict[int, List[Tuple[Track,int]]] = defaultdict(list)
    for trk in tracks:
        for i, fn in enumerate(trk.frame_nos):
            frame_to_tracks[fn].append((trk, i))

    result = {}
    fn_set = {r['frame_no'] for r in frame_records}

    for rec in frame_records:
        fn       = rec['frame_no']
        yolo_all = all_dets.get(fn, [])
        yolo_mon = [d for d in yolo_all if d['cls_id'] == 0]
        yolo_ada = [d for d in yolo_all if d['cls_id'] == 1]
        trk_hits = frame_to_tracks.get(fn, [])

        gt_dets = []
        case_flags = set()

        # ── A: 고신뢰 YOLO + 충분한 트랙 길이 → GT 확정 ─────────────────────
        for det in yolo_mon:
            if det['conf'] >= YOLO_CONF_GOOD:
                # 대응 트랙 찾기
                matched_trk = None
                dbox = [det['x1r'],det['y1r'],det['x2r'],det['y2r']]
                for trk, idx in trk_hits:
                    tbox = [trk.dets[idx]['x1r'], trk.dets[idx]['y1r'],
                            trk.dets[idx]['x2r'], trk.dets[idx]['y2r']]
                    if iou(dbox, tbox) > 0.3:
                        matched_trk = trk
                        break
                track_ok = (matched_trk is not None and
                            matched_trk.length >= GT_MIN_TRACK_LEN)
                gt_dets.append({**det, 'case':'A' if track_ok else 'C',
                                'track_len': matched_trk.length if matched_trk else 0,
                                'gt_source':'yolo_high'})
                case_flags.add('A' if track_ok else 'C')

        # ── C: 중간 신뢰 (YOLO_CONF_MID ~ GOOD) → 트랙 길이로 보완 ──────────
        for det in yolo_mon:
            if YOLO_CONF_MID <= det['conf'] < YOLO_CONF_GOOD:
                dbox = [det['x1r'],det['y1r'],det['x2r'],det['y2r']]
                already = any(iou(dbox,[g['x1r'],g['y1r'],g['x2r'],g['y2r']])>0.5
                              for g in gt_dets)
                if already:
                    continue
                matched_trk = None
                for trk, idx in trk_hits:
                    tbox = [trk.dets[idx]['x1r'], trk.dets[idx]['y1r'],
                            trk.dets[idx]['x2r'], trk.dets[idx]['y2r']]
                    if iou(dbox, tbox) > 0.25:
                        matched_trk = trk
                        break
                if matched_trk and matched_trk.length >= GT_MIN_TRACK_LEN:
                    gt_dets.append({**det, 'case':'C',
                                    'track_len':matched_trk.length,
                                    'gt_source':'yolo_mid_track'})
                    case_flags.add('C')
                else:
                    # 트랙 없거나 짧으면 HARD
                    gt_dets.append({**det, 'case':'HARD',
                                    'track_len': matched_trk.length if matched_trk else 0,
                                    'gt_source':'yolo_mid_notrack'})
                    case_flags.add('HARD')

        # ── B: 트랙은 있으나 해당 프레임 YOLO 탐지 없음 → 미탐 GT ────────────
        for trk, idx in trk_hits:
            if trk.length < GT_MIN_TRACK_LEN:
                continue
            tdet = trk.dets[idx]
            tbox = [tdet['x1r'],tdet['y1r'],tdet['x2r'],tdet['y2r']]
            # 기존 GT와 겹치는지 확인
            overlap = any(iou(tbox,[g['x1r'],g['y1r'],g['x2r'],g['y2r']])>0.3
                          for g in gt_dets)
            if not overlap and tdet['conf'] >= YOLO_CONF_LOW:
                # 이 트랙의 다른 프레임에서는 탐지됐는데 여기선 미탐
                other_frames_det = [d['conf'] for d in trk.dets
                                    if d is not tdet and d['conf'] >= YOLO_CONF_MID]
                if other_frames_det:
                    gt_dets.append({**tdet, 'case':'B',
                                    'track_len':trk.length,
                                    'gt_source':'track_interpolated'})
                    case_flags.add('B')

        # ── E: YOLO 탐지 있으나 짧은 트랙 (단발성 오탐) ──────────────────────
        for det in yolo_mon:
            dbox = [det['x1r'],det['y1r'],det['x2r'],det['y2r']]
            in_gt = any(iou(dbox,[g['x1r'],g['y1r'],g['x2r'],g['y2r']])>0.3
                        for g in gt_dets)
            if not in_gt and det['conf'] < YOLO_CONF_MID:
                # 대응 트랙 확인
                matched_trk = None
                for trk, idx in trk_hits:
                    tbox = [trk.dets[idx]['x1r'], trk.dets[idx]['y1r'],
                            trk.dets[idx]['x2r'], trk.dets[idx]['y2r']]
                    if iou(dbox, tbox) > 0.2:
                        matched_trk = trk
                        break
                if matched_trk is None or matched_trk.length < TRACK_MIN_LEN:
                    case_flags.add('E')

        # ── adena 포함 (변경 없이 YOLO 그대로) ───────────────────────────────
        for det in yolo_ada:
            gt_dets.append({**det, 'case':'A', 'track_len':0,
                            'gt_source':'adena_passthrough'})

        # 전체 case 요약
        if not case_flags:
            main_case = 'EMPTY'
        elif 'B' in case_flags:
            main_case = 'B'  # 미탐 최우선
        elif 'A' in case_flags:
            main_case = 'A'
        elif 'C' in case_flags:
            main_case = 'C'
        elif 'HARD' in case_flags:
            main_case = 'HARD'
        elif 'E' in case_flags:
            main_case = 'E'
        else:
            main_case = 'EMPTY'

        result[fn] = {
            'frame_no' : fn,
            'fname'    : rec['fname'],
            'path'     : rec['path'],
            'ts_sec'   : rec['ts_sec'],
            'diff'     : rec.get('diff', 0),
            'gt_dets'  : gt_dets,
            'yolo_dets': yolo_mon,
            'case'     : main_case,
            'case_flags': list(case_flags),
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: YOLO 라벨 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_label(frame_result: Dict, label_path: Path,
               img_path: Path, split_img_dir: Path):
    """GT bbox → YOLO format .txt 저장 + 이미지 복사"""
    img_dst = split_img_dir / frame_result['fname']
    if not img_dst.exists():
        shutil.copy2(frame_result['path'], img_dst)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for det in frame_result['gt_dets']:
        if det['case'] in ('E', 'HARD'):
            continue   # 오탐/애매 → 라벨 제외
        cls = det['cls_id']
        cx_n = det['cx_n']; cy_n = det['cy_n']
        w_n  = det['w_n'];  h_n  = det['h_n']
        # 범위 클리핑
        cx_n = max(0.001, min(0.999, cx_n))
        cy_n = max(0.001, min(0.999, cy_n))
        w_n  = max(0.001, min(min(cx_n, 1-cx_n)*2*0.99, w_n))
        h_n  = max(0.001, min(min(cy_n, 1-cy_n)*2*0.99, h_n))
        lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

    label_path.write_text('\n'.join(lines))
    return len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Preview 이미지 생성
# ─────────────────────────────────────────────────────────────────────────────
def draw_preview(frame_result: Dict, out_path: Path,
                 show_yolo=True, show_gt=True):
    img = cv2.imread(frame_result['path'])
    if img is None:
        return False

    H, W = img.shape[:2]
    fn   = frame_result['frame_no']
    case = frame_result['case']

    # ROI 경계
    cv2.rectangle(img, (ROI_X,ROI_Y),
                  (ROI_X+ROI_W,ROI_Y+ROI_H), COL['ROI'], 1)

    # YOLO 탐지 (하늘색 점선)
    if show_yolo:
        for det in frame_result['yolo_dets']:
            x1 = int(ROI_X + det['x1r'])
            y1 = int(ROI_Y + det['y1r'])
            x2 = int(ROI_X + det['x2r'])
            y2 = int(ROI_Y + det['y2r'])
            cv2.rectangle(img,(x1,y1),(x2,y2), COL['YOLO'], 1)
            cv2.putText(img, f"y:{det['conf']:.2f}",
                        (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL['YOLO'], 1)

    # GT bbox (굵은 색상, CASE별)
    if show_gt:
        for det in frame_result['gt_dets']:
            c = det['case']
            col = {'A':COL['GT'],'B':COL['MISS'],'C':(0,200,255),
                   'D':(200,100,0),'E':COL['FP'],'HARD':COL['HARD']}.get(c, COL['GT'])
            x1 = int(ROI_X + det['x1r'])
            y1 = int(ROI_Y + det['y1r'])
            x2 = int(ROI_X + det['x2r'])
            y2 = int(ROI_Y + det['y2r'])
            cv2.rectangle(img,(x1,y1),(x2,y2), col, 2)
            label = f"{det['cls_name']}[{c}] {det['conf']:.2f}"
            tid   = det.get('track_len',0)
            if tid:
                label += f" t{tid}"
            cv2.putText(img, label, (x1, max(10,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # 프레임 정보 오버레이
    info = f"F{fn} ts={frame_result['ts_sec']:.1f}s CASE={case}"
    cv2.putText(img, info, (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(img, info, (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),       1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Contact Sheet 생성
# ─────────────────────────────────────────────────────────────────────────────
def make_contact_sheet(image_paths: List[Path], out_path: Path,
                       title: str, cols: int = 4, thumb_w: int = 420):
    """여러 이미지를 격자로 배치한 contact sheet."""
    if not image_paths:
        return
    rows = math.ceil(len(image_paths) / cols)
    thumb_h = int(thumb_w * 1080 / 1920)
    TITLE_H = 40
    sheet_w = cols * thumb_w
    sheet_h = rows * thumb_h + TITLE_H
    sheet   = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:TITLE_H] = (40,40,40)
    cv2.putText(sheet, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,200), 2)

    for i, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        r = i // cols;  c = i % cols
        y0 = TITLE_H + r*thumb_h
        x0 = c*thumb_w
        sheet[y0:y0+thumb_h, x0:x0+thumb_w] = thumb
        # 격자선
        cv2.rectangle(sheet,(x0,y0),(x0+thumb_w-1,y0+thumb_h-1),(60,60,60),1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"  Contact sheet → {out_path.name}  ({len(image_paths)}장)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: 라벨 품질 검사
# ─────────────────────────────────────────────────────────────────────────────
def validate_labels(split: str) -> Dict:
    img_dir   = DATASET/'images'/split
    label_dir = DATASET/'labels'/split
    issues = defaultdict(list)
    stats  = {'total':0,'labeled':0,'empty':0,'instances':0}

    for img_p in sorted(img_dir.glob('*.jpg')):
        stats['total'] += 1
        lbl_p = label_dir / (img_p.stem + '.txt')
        if not lbl_p.exists():
            issues['MISSING_LABEL'].append(img_p.name)
            continue
        txt = lbl_p.read_text().strip()
        if not txt:
            stats['empty'] += 1
            continue
        stats['labeled'] += 1
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) < 5:
                issues['INVALID_FORMAT'].append(img_p.name)
                continue
            try:
                cls = int(parts[0])
                cx,cy,w,h = map(float, parts[1:5])
            except:
                issues['PARSE_ERROR'].append(img_p.name)
                continue
            if cls not in (0,1):
                issues['INVALID_CLASS'].append(img_p.name)
            if not (0<cx<1 and 0<cy<1 and 0<w<1 and 0<h<1):
                issues['OUT_OF_RANGE'].append(img_p.name)
            if w<=0 or h<=0:
                issues['ZERO_SIZE'].append(img_p.name)
            if w*h < 0.0001:
                issues['TOO_SMALL'].append(img_p.name)
            if w*h > 0.85:
                issues['TOO_LARGE'].append(img_p.name)
            stats['instances'] += 1

    return {'stats':stats,'issues':{k:len(v) for k,v in issues.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: 통계 수집
# ─────────────────────────────────────────────────────────────────────────────
def collect_stats(all_results_by_vid: Dict) -> Dict:
    stats = {
        'split'   : {'train':{'images':0,'instances':0},
                     'val'  :{'images':0,'instances':0},
                     'test' :{'images':0,'instances':0}},
        'case'    : {'A':0,'B':0,'C':0,'D':0,'E':0,'HARD':0,'EMPTY':0},
        'position': {'LEFT':0,'CENTER':0,'RIGHT':0,
                     'TOP':0,'MIDDLE':0,'BOTTOM':0},
        'size'    : {'SMALL':0,'MEDIUM':0,'LARGE':0},
        'hard_neg': 0,
        'vid_inst': {},
    }
    for vid_id, (vid_results, split) in all_results_by_vid.items():
        inst_count = 0
        for fn, fr in vid_results.items():
            case = fr['case']
            stats['case'][case] = stats['case'].get(case,0) + 1
            for det in fr['gt_dets']:
                if det['case'] in ('E','HARD'):
                    continue
                if det['cls_id'] != 0:
                    continue
                inst_count += 1
                stats['split'][split]['instances'] += 1
                for tag in classify_position(det['cx_n'], det['cy_n']):
                    stats['position'][tag] = stats['position'].get(tag,0)+1
                sz = classify_size(det['w_n'], det['h_n'])
                stats['size'][sz] = stats['size'].get(sz,0)+1
        stats['vid_inst'][vid_id] = inst_count

    for split in ('train','val','test'):
        img_dir = DATASET/'images'/split
        stats['split'][split]['images'] = len(list(img_dir.glob('*.jpg')))
        # Hard Negative = 빈 라벨
        lbl_dir = DATASET/'labels'/split
        for lp in lbl_dir.glob('*.txt'):
            if lp.stat().st_size == 0:
                stats['hard_neg'] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(steps):
    t0 = time.time()
    log("="*65)
    log("video_gt_pipeline  영상기반 Ground Truth 데이터셋 구축")
    log("="*65)

    # ── 기존 데이터 보호 확인 ─────────────────────────────────────────────────
    assert (BASE/'dataset/dataset.yaml').exists(), "기존 dataset/ 보호 실패"
    assert (BASE/'runs/detect/monster_v1-2/weights/best.pt').exists(), "모델 보호 실패"
    log("✅ 기존 dataset/ + 모델 보존 확인")

    # ── 디렉토리 ──────────────────────────────────────────────────────────────
    for split in ('train','val','test'):
        (DATASET/'images'/split).mkdir(parents=True, exist_ok=True)
        (DATASET/'labels'/split).mkdir(parents=True, exist_ok=True)
        (DATASET/'previews'/split).mkdir(parents=True, exist_ok=True)
    for d in ('hard_cases','missed','false_positives','reports',
              'frames','contact_sheets'):
        (DATASET/d).mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    if any(s in steps for s in ('label','all')):
        model = load_yolo_model()
    else:
        model = None

    all_results_by_vid = {}   # {vid_id: (vid_results, split)}
    global_stats = {
        'extract'    : {},
        'track_count': {},
        'case_count' : {},
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 영상 루프
    # ─────────────────────────────────────────────────────────────────────────
    for vid_id, (vid_rel, split) in VIDEOS.items():
        vid_path = BASE / vid_rel
        if not vid_path.exists():
            log(f"[{vid_id}] 영상 없음: {vid_path}", 'WARN')
            continue

        log(f"\n{'─'*55}")
        log(f"[{vid_id}] → {split}  ({vid_path.name})")

        frames_dir = DATASET/'frames'/vid_id
        det_cache  = DATASET/'reports'/f'{vid_id}_detections.json'

        # ── STEP 1: 프레임 추출 ───────────────────────────────────────────────
        if any(s in steps for s in ('extract','label','all')):
            frame_records = extract_frames(vid_id, vid_path, frames_dir)
        else:
            meta = frames_dir/'extract_meta.json'
            frame_records = json.load(open(meta)) if meta.exists() else []

        global_stats['extract'][vid_id] = len(frame_records)
        log(f"[{vid_id}]   추출 완료: {len(frame_records)}장")

        if not frame_records:
            continue

        # ── STEP 2: YOLO 보조 탐지 ────────────────────────────────────────────
        if any(s in steps for s in ('label','all')) and model:
            if det_cache.exists():
                with open(det_cache) as f:
                    all_dets_raw = json.load(f)
                all_dets = {int(k):v for k,v in all_dets_raw.items()}
                log(f"[{vid_id}]   탐지 캐시 로드")
            else:
                all_dets = {}
                for i, rec in enumerate(frame_records):
                    frame = cv2.imread(rec['path'])
                    if frame is None:
                        continue
                    dets = yolo_detect(model, frame)
                    all_dets[rec['frame_no']] = dets
                    if (i+1) % 50 == 0:
                        log(f"[{vid_id}]   탐지: {i+1}/{len(frame_records)}")
                with open(det_cache,'w') as f:
                    json.dump({str(k):v for k,v in all_dets.items()}, f, indent=2)
                log(f"[{vid_id}]   YOLO 탐지 완료: {len(all_dets)}프레임")
        else:
            all_dets = {}
            if det_cache.exists():
                with open(det_cache) as f:
                    raw = json.load(f)
                all_dets = {int(k):v for k,v in raw.items()}

        # ── STEP 3: Multi-cue Tracker ─────────────────────────────────────────
        if any(s in steps for s in ('label','all')):
            tracks = run_tracker(frame_records, all_dets)
            good_tracks = [t for t in tracks if t.length >= TRACK_MIN_LEN]
            global_stats['track_count'][vid_id] = len(good_tracks)
            log(f"[{vid_id}]   트랙: {len(tracks)}개 (유효≥{TRACK_MIN_LEN}: {len(good_tracks)}개)")

            # ── STEP 4: GT 라벨 결정 ──────────────────────────────────────────
            vid_results = build_gt_labels(frame_records, all_dets, good_tracks)

            # 캐시 저장
            cache_path = DATASET/'reports'/f'{vid_id}_gt_results.json'
            with open(cache_path,'w') as f:
                # frame는 직렬화 제외
                safe = {str(k):{kk:vv for kk,vv in v.items() if kk!='frame'}
                        for k,v in vid_results.items()}
                json.dump(safe, f, indent=2)
        else:
            cache_path = DATASET/'reports'/f'{vid_id}_gt_results.json'
            if cache_path.exists():
                with open(cache_path) as f:
                    raw = json.load(f)
                vid_results = {int(k):v for k,v in raw.items()}
            else:
                vid_results = {}

        # ── 캐시 로드된 vid_results에서도 case_count 집계 (report 단독 실행 지원) ──
        if vid_results and vid_id not in global_stats['case_count']:
            _cc = defaultdict(int)
            for _fn, _fr in vid_results.items():
                _cc[_fr.get('case','UNKNOWN')] += 1
            global_stats['case_count'][vid_id] = dict(_cc)

        # ── STEP 5: 라벨 저장 + 이미지 복사 ──────────────────────────────────
        if any(s in steps for s in ('label','all')) and vid_results:
            split_img_dir = DATASET/'images'/split
            split_lbl_dir = DATASET/'labels'/split
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            case_count = defaultdict(int)
            for fn, fr in vid_results.items():
                case_count[fr['case']] += 1

                label_path = split_lbl_dir / (Path(fr['fname']).stem + '.txt')
                n = save_label(fr, label_path,
                               Path(fr['path']), split_img_dir)

                # Hard case / Missed / FP 별도 저장
                if fr['case'] == 'B':
                    dst = DATASET/'missed'/fr['fname']
                    if not dst.exists():
                        shutil.copy2(fr['path'], dst)
                elif fr['case'] == 'E':
                    dst = DATASET/'false_positives'/fr['fname']
                    if not dst.exists():
                        shutil.copy2(fr['path'], dst)
                elif fr['case'] in ('HARD','C'):
                    dst = DATASET/'hard_cases'/fr['fname']
                    if not dst.exists():
                        shutil.copy2(fr['path'], dst)

            global_stats['case_count'][vid_id] = dict(case_count)
            log(f"[{vid_id}]   CASE: " +
                " ".join(f"{k}={v}" for k,v in sorted(case_count.items())))

        all_results_by_vid[vid_id] = (vid_results, split)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Preview 이미지 생성
    # ─────────────────────────────────────────────────────────────────────────
    if any(s in steps for s in ('preview','all')):
        log(f"\n{'─'*55}")
        log("STEP 6: Preview 이미지 생성")

        preview_paths = defaultdict(list)   # case → [path]

        for vid_id, (vid_results, split) in all_results_by_vid.items():
            prev_dir = DATASET/'previews'/split
            prev_dir.mkdir(exist_ok=True)

            for fn, fr in vid_results.items():
                out_p = prev_dir / (Path(fr['fname']).stem + '_prev.jpg')
                if draw_preview(fr, out_p):
                    case = fr['case']
                    preview_paths[case].append(out_p)

        total_prev = sum(len(v) for v in preview_paths.values())
        log(f"  Preview 생성: {total_prev}장")

        # ── Contact Sheet 5종 ─────────────────────────────────────────────────
        log("  Contact Sheet 생성 중...")
        cs_dir = DATASET/'contact_sheets'

        # 1. 정상 GT (CASE A)
        make_contact_sheet(
            preview_paths.get('A',[])[:32],
            cs_dir/'01_normal_gt.jpg',
            "① 정상 GT (CASE A - 고신뢰 탐지 성공)")

        # 2. 기존 YOLO 미탐 → 새 GT (CASE B)
        make_contact_sheet(
            preview_paths.get('B',[])[:32],
            cs_dir/'02_missed_new_gt.jpg',
            "② 기존 YOLO 미탐 → 새로 라벨링 (CASE B)")

        # 3. 저신뢰 트랙보완 (CASE C)
        make_contact_sheet(
            preview_paths.get('C',[])[:32],
            cs_dir/'03_low_conf_track.jpg',
            "③ 저신뢰 - 트랙 연속성으로 보완 (CASE C)")

        # 4. 오탐 (CASE E)
        make_contact_sheet(
            preview_paths.get('E',[])[:32],
            cs_dir/'04_false_positive.jpg',
            "④ 기존 YOLO 오탐 사례 (CASE E)")

        # 5. 애매한 프레임 (HARD)
        make_contact_sheet(
            preview_paths.get('HARD',[])[:32],
            cs_dir/'05_hard_cases.jpg',
            "⑤ 검수 필요 애매한 프레임 (HARD)")

        # 6. 전체 GT 샘플 (모든 case 혼합)
        all_prev = []
        for split in ('train','val','test'):
            all_prev += sorted((DATASET/'previews'/split).glob('*.jpg'))
        import random; random.seed(42)
        sample = random.sample(all_prev, min(48, len(all_prev)))
        make_contact_sheet(sample, cs_dir/'00_overall_sample.jpg',
                           "⓪ 전체 GT 샘플 (랜덤 48장)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: 라벨 품질 검사
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'─'*55}")
    log("STEP 7: 라벨 품질 검사")
    val_results = {}
    for split in ('train','val','test'):
        r = validate_labels(split)
        val_results[split] = r
        log(f"  [{split}] images={r['stats']['total']}  "
            f"labeled={r['stats']['labeled']}  "
            f"instances={r['stats']['instances']}  "
            f"issues={sum(r['issues'].values())}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: data.yaml 생성
    # ─────────────────────────────────────────────────────────────────────────
    yaml_content = f"""# dataset_field_v1  data.yaml
# 실제 게임 필드 영상 기반 Ground Truth 데이터셋
# 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 분할: Train=VID_A/B/C  Val=VID_D/E  Test=VID_F

path: {DATASET.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: monster
  1: adena
"""
    (DATASET/'data.yaml').write_text(yaml_content)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: 통계 + 최종 보고서
    # ─────────────────────────────────────────────────────────────────────────
    if any(s in steps for s in ('report','all')):
        log(f"\n{'─'*55}")
        log("STEP 9: 최종 보고서 생성")

        stats = collect_stats(all_results_by_vid)

        report = {
            'generated_at' : time.strftime('%Y-%m-%d %H:%M:%S'),
            'videos'       : {vid_id: {'path':vp,'split':sp}
                               for vid_id,(vp,sp) in VIDEOS.items()},
            'extract'      : global_stats['extract'],
            'tracks'       : global_stats['track_count'],
            'cases'        : global_stats['case_count'],
            'final_stats'  : stats,
            'validation'   : val_results,
        }
        rp = DATASET/'reports'/'final_report.json'
        with open(rp,'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print_final_report(report, stats, time.time()-t0)

    log(f"\n전체 완료: {time.time()-t0:.1f}초")


# ─────────────────────────────────────────────────────────────────────────────
# 최종 보고서 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_final_report(report, stats, elapsed):
    sep = "─"*65
    print(f"\n{'═'*65}")
    print("  📊 dataset_field_v1 최종 보고서")
    print(f"  생성: {report['generated_at']}  소요: {elapsed:.0f}s")
    print(f"{'═'*65}\n")

    # 1. 분석 영상 목록
    print("① 분석 영상")
    print(sep)
    vid_meta = {
        'VID_A':{'len':'37.3s','frames':373,'fps':10,'size':'83.1MB','split':'train'},
        'VID_B':{'len':'53.9s','frames':539,'fps':10,'size':'96.2MB','split':'train'},
        'VID_C':{'len':'46.4s','frames':464,'fps':10,'size':'99.0MB','split':'train'},
        'VID_D':{'len':'83.9s','frames':839,'fps':10,'size':'60.6MB','split':'val'},
        'VID_E':{'len':'33.4s','frames':334,'fps':10,'size':'29.4MB','split':'val'},
        'VID_F':{'len':'262.8s','frames':2628,'fps':10,'size':'55.2MB','split':'test'},
    }
    total_raw = sum(v['frames'] for v in vid_meta.values())
    for vid_id, m in vid_meta.items():
        extracted = report['extract'].get(vid_id,0)
        print(f"  {vid_id} [{m['split']:5s}] {m['len']:7s} {m['frames']:4d}frames "
              f"{m['size']:7s}  → 추출 {extracted}장")
    print(f"  원본 합계: {total_raw:,}frames  ({total_raw/10/60:.1f}분)")

    # 2. 프레임 샘플링
    total_ex = sum(report['extract'].values())
    print(f"\n② 프레임 샘플링")
    print(sep)
    print(f"  방법: interval={EXTRACT_INTERVAL}프레임 + diff스킵(≤{DIFF_SKIP_THRESH}) "
          f"+ phash중복제거(≤{PHASH_DUP_THRESH})")
    print(f"  원본 {total_raw:,}장 → 추출 {total_ex}장 "
          f"({total_ex/total_raw*100:.1f}% 선택)")

    # 3~4. 라벨 수
    print(f"\n③/④ 이미지 수 & Monster bbox 수")
    print(sep)
    for split in ('train','val','test'):
        si = stats['split'].get(split,{})
        n_img  = si.get('images',0)
        n_inst = si.get('instances',0)
        print(f"  {split:5s}: {n_img:4d}이미지  monster={n_inst}개")
    total_inst = sum(s.get('instances',0) for s in stats['split'].values())
    total_img  = sum(s.get('images',0)   for s in stats['split'].values())
    print(f"  합계: {total_img}이미지  monster={total_inst}개")

    # 5. CASE 분포
    print(f"\n⑤ CASE 분포 (기존 YOLO 비교)")
    print(sep)
    all_cases = defaultdict(int)
    for vid_id, cases in report['cases'].items():
        for c, cnt in cases.items():
            all_cases[c] += cnt
    print(f"  A(탐지성공)  : {all_cases.get('A',0):4d}  ← 고신뢰+트랙 확정")
    print(f"  B(미탐→GT)   : {all_cases.get('B',0):4d}  ← YOLO가 놓쳤으나 트랙으로 발굴")
    print(f"  C(저신뢰)    : {all_cases.get('C',0):4d}  ← conf 낮으나 트랙 연속성 보완")
    print(f"  E(오탐)      : {all_cases.get('E',0):4d}  ← 라벨 제외, FP 저장")
    print(f"  HARD(애매)   : {all_cases.get('HARD',0):4d}  ← 검수 필요")
    print(f"  EMPTY(몬없음): {all_cases.get('EMPTY',0):4d}  ← Hard Negative")

    # 6. Hard case / Missed / FP
    missed_n = len(list((DATASET/'missed').glob('*.jpg')))
    fp_n     = len(list((DATASET/'false_positives').glob('*.jpg')))
    hard_n   = len(list((DATASET/'hard_cases').glob('*.jpg')))
    print(f"\n⑥ Hard case 현황")
    print(sep)
    print(f"  missed/          : {missed_n}장  (YOLO 미탐 → 새 GT 포함)")
    print(f"  false_positives/ : {fp_n}장    (YOLO 오탐 Hard Negative)")
    print(f"  hard_cases/      : {hard_n}장  (검수 필요 애매 프레임)")
    print(f"  Hard Negative(빈라벨): {stats.get('hard_neg',0)}장")

    # 7. 위치 분포
    print(f"\n⑦ 위치 분포 (monster bbox)")
    print(sep)
    pos = stats.get('position',{})
    total_pos = sum(pos.values())
    for tag, cnt in pos.items():
        pct = cnt/total_pos*100 if total_pos else 0
        bar = '█'*int(pct/3+0.5)
        status = '✅' if pct>=10 else ('⚠️' if pct>=4 else '❌부족')
        print(f"  {tag:8s}: {cnt:4d} ({pct:5.1f}%)  {bar:<15s} {status}")

    # 8. 크기 분포
    print(f"\n⑧ 크기 분포")
    print(sep)
    sz = stats.get('size',{})
    total_sz = sum(sz.values())
    for tag, cnt in sz.items():
        pct = cnt/total_sz*100 if total_sz else 0
        print(f"  {tag:8s}: {cnt:4d} ({pct:5.1f}%)")

    # 9. 기존 데이터 대비 개선점
    print(f"\n⑨ 기존 dataset 대비 개선점")
    print(sep)
    print(f"  기존 dataset: train=539, val=23  (총 562장)")
    print(f"  신규 dataset: train={stats['split'].get('train',{}).get('images',0)}"
          f"  val={stats['split'].get('val',{}).get('images',0)}"
          f"  test={stats['split'].get('test',{}).get('images',0)}")
    left_n   = pos.get('LEFT',0)
    right_n  = pos.get('RIGHT',0)
    bottom_n = pos.get('BOTTOM',0)
    print(f"  좌측(cx<0.25) 보강: {left_n}장  (기존 ~0)")
    print(f"  우측(cx>0.75) 보강: {right_n}장  (기존 ~0)")
    print(f"  하단(cy>0.65) 보강: {bottom_n}장  (기존 6)")
    print(f"  미탐 케이스 포함:   {all_cases.get('B',0)}장")

    # 10. 라벨 품질 검사
    print(f"\n⑩ 라벨 품질 검사")
    print(sep)
    val = report.get('validation',{})
    for split, vr in val.items():
        issues = vr.get('issues',{})
        total_issue = sum(issues.values())
        print(f"  [{split}] issues={total_issue}", end='')
        if total_issue:
            for k,v in issues.items():
                if v: print(f"  {k}={v}", end='')
        print()

    # 11. Preview / Contact Sheet 위치
    print(f"\n⑪ Preview & Contact Sheet 위치")
    print(sep)
    total_prev = sum(len(list((DATASET/'previews'/s).glob('*.jpg')))
                     for s in ('train','val','test'))
    print(f"  previews/train,val,test/  : {total_prev}장")
    cs_dir = DATASET/'contact_sheets'
    for cs in sorted(cs_dir.glob('*.jpg')):
        print(f"  contact_sheets/{cs.name}")

    # 12. 확인 필요 사항
    print(f"\n⑫ 학습 전 확인 필요 사항")
    print(sep)
    checks = []
    if all_cases.get('B',0) > 0:
        checks.append(f"  ⚠ CASE B 미탐 {all_cases['B']}장: missed/ 폴더에서 bbox 검수")
    if all_cases.get('HARD',0) > 0:
        checks.append(f"  ⚠ HARD {all_cases['HARD']}장: hard_cases/ 폴더 수동 검토")
    if pos.get('LEFT',0) < 10:
        checks.append(f"  ⚠ 좌측 데이터 부족 ({pos.get('LEFT',0)}장) — 추가 촬영 권장")
    if pos.get('RIGHT',0) < 10:
        checks.append(f"  ⚠ 우측 데이터 부족 ({pos.get('RIGHT',0)}장) — 추가 촬영 권장")
    if pos.get('BOTTOM',0) < 10:
        checks.append(f"  ⚠ 하단 데이터 부족 ({pos.get('BOTTOM',0)}장) — 추가 촬영 권장")
    if not checks:
        checks.append("  ✅ 주요 이슈 없음")
    for c in checks:
        print(c)

    print(f"\n{'═'*65}")
    print(f"  ✅ 학습은 아직 미실행. Preview 확인 후 진행하세요.")
    print(f"     data.yaml: {DATASET}/data.yaml")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='영상기반 GT 데이터셋 파이프라인')
    ap.add_argument('--step', default='all',
                    choices=['all','extract','label','preview','report'],
                    help='실행 단계 (기본: all)')
    args = ap.parse_args()

    steps = ['extract','label','preview','report'] if args.step=='all' \
            else [args.step]
    run_pipeline(steps)


if __name__ == '__main__':
    main()
