"""
tools/filter_hard_cases.py
==========================
자동 라벨링 결과를 분석해서 다음을 분류한다.

CASE A  실제 몬스터 있음 + YOLO 탐지 성공 (GOOD)
CASE B  실제 몬스터 있음 + YOLO 미탐        (최우선 라벨링 후보)
CASE C  실제 몬스터 있음 + bbox 불량         (검수 필요)
CASE D  실제 몬스터 없음 + YOLO 오탐         (Hard Negative)

연속 프레임 컨텍스트를 활용해 CASE B를 자동 추론한다:
  t-2/t-1 에서 탐지됐고 t+1/t+2 에서 탐지되면 → t 에서 미탐 = CASE B 후보

출력:
  hard_cases/  - CASE B/C/D 이미지 + 메타
  missed/      - CASE B 이미지 (미탐 최우선)
  false_positives/ - CASE D 이미지

사용법:
  python tools/filter_hard_cases.py \
      --det  dataset_field_v1/labels_auto/detection_results.json \
      --frames dataset_field_v1/frames \
      --out   dataset_field_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── ROI 설정 ─────────────────────────────────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472

# ── 연속 프레임 윈도우 (미탐 추론) ──────────────────────────────────────────
CONTEXT_WINDOW = 3   # 앞뒤 N프레임 확인

# ── confidence 임계값 ─────────────────────────────────────────────────────────
CONF_GOOD = 0.50
CONF_MID  = 0.20
CONF_LOW  = 0.10

# ── bbox 품질 판단 ─────────────────────────────────────────────────────────
MIN_AREA  = 200    # px² 이하는 너무 작음
MAX_RATIO = 5.0    # w/h 또는 h/w 이상이면 비정상


def classify_bbox_quality(det: Dict) -> str:
    """
    탐지된 bbox 품질 판단.
    GOOD   : 정상
    SMALL  : 너무 작음
    CLIPPED: ROI 경계 걸림
    ODD    : 비율 이상
    """
    w, h = det['w_abs'], det['h_abs']
    if det['clipped']:
        return 'CLIPPED'
    if w * h < MIN_AREA:
        return 'SMALL'
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > MAX_RATIO:
        return 'ODD'
    return 'GOOD'


# ══════════════════════════════════════════════════════════════════════════════
# 연속 프레임 컨텍스트 기반 미탐 탐지
# ══════════════════════════════════════════════════════════════════════════════

def find_missed_by_context(records: List[Dict]) -> List[Dict]:
    """
    연속 프레임 컨텍스트로 CASE B (미탐) 후보 찾기.

    알고리즘:
      각 프레임 i에 monster 탐지가 없을 때,
      [i-CONTEXT_WINDOW, i-1]에 탐지가 있고
      [i+1, i+CONTEXT_WINDOW]에도 탐지가 있으면
      → i 프레임은 CASE B 후보

    단, 영상이 달라지는 경계에서는 적용하지 않음 (video_id 확인).
    """
    # 영상 ID별로 그룹화
    by_video = defaultdict(list)
    for rec in records:
        vid = rec.get('video_id', 'UNK')
        by_video[vid].append(rec)

    missed_candidates = []

    for vid, vid_records in by_video.items():
        # frame_no 순 정렬
        sorted_recs = sorted(vid_records, key=lambda r: r.get('frame_no', 0))
        n = len(sorted_recs)

        for i, rec in enumerate(sorted_recs):
            # 이미 탐지된 프레임 → 스킵
            monsters = [d for d in rec['detections'] if d['class_id'] == 0]
            if monsters:
                continue

            # 앞 CONTEXT_WINDOW 프레임에 탐지 있는지 확인
            prev_has_det = False
            for j in range(max(0, i - CONTEXT_WINDOW), i):
                prev_mons = [d for d in sorted_recs[j]['detections']
                             if d['class_id'] == 0 and d['conf'] >= CONF_MID]
                if prev_mons:
                    prev_has_det = True
                    break

            if not prev_has_det:
                continue

            # 뒤 CONTEXT_WINDOW 프레임에 탐지 있는지 확인
            next_has_det = False
            for j in range(i + 1, min(n, i + CONTEXT_WINDOW + 1)):
                next_mons = [d for d in sorted_recs[j]['detections']
                             if d['class_id'] == 0 and d['conf'] >= CONF_MID]
                if next_mons:
                    next_has_det = True
                    break

            if prev_has_det and next_has_det:
                # 전후 프레임에서 monster bbox 참조
                ref_dets = []
                for j in range(max(0, i - CONTEXT_WINDOW), i):
                    m = [d for d in sorted_recs[j]['detections']
                         if d['class_id'] == 0]
                    if m:
                        ref_dets.extend(m)

                missed_candidates.append({
                    'case':      'B',
                    'reason':    '연속 프레임 컨텍스트 미탐',
                    'fname':     rec['fname'],
                    'path':      rec['path'],
                    'video_id':  vid,
                    'frame_no':  rec.get('frame_no', -1),
                    'ref_dets':  ref_dets[:3],  # 참조용 최대 3건
                })

    return missed_candidates


# ══════════════════════════════════════════════════════════════════════════════
# 오탐 후보 (CASE D)
# ══════════════════════════════════════════════════════════════════════════════

def find_false_positives(records: List[Dict]) -> List[Dict]:
    """
    CASE D: YOLO가 monster라고 했는데 오탐 가능성이 높은 케이스.
    - 연속 프레임에서 갑자기 등장했다 사라지는 단발 탐지 (1~2프레임만)
    - confidence < 0.20 인 탐지
    """
    by_video = defaultdict(list)
    for rec in records:
        vid = rec.get('video_id', 'UNK')
        by_video[vid].append(rec)

    fp_candidates = []

    for vid, vid_records in by_video.items():
        sorted_recs = sorted(vid_records, key=lambda r: r.get('frame_no', 0))
        n = len(sorted_recs)

        for i, rec in enumerate(sorted_recs):
            monsters = [d for d in rec['detections']
                        if d['class_id'] == 0 and d['conf'] >= CONF_LOW]
            if not monsters:
                continue

            for det in monsters:
                reason = None

                # 저신뢰 탐지 (FP 가능)
                if det['conf'] < CONF_MID:
                    reason = f"저신뢰(conf={det['conf']:.3f})"

                # 단발 탐지 (앞뒤 없음) → FP 가능
                else:
                    prev_has = any(
                        any(d['class_id'] == 0 for d in sorted_recs[j]['detections'])
                        for j in range(max(0, i - 2), i)
                    ) if i > 0 else False

                    next_has = any(
                        any(d['class_id'] == 0 for d in sorted_recs[j]['detections'])
                        for j in range(i + 1, min(n, i + 3))
                    ) if i < n - 1 else False

                    if not prev_has and not next_has:
                        reason = "단발 탐지 (앞뒤 없음)"

                if reason:
                    fp_candidates.append({
                        'case':     'D',
                        'reason':   reason,
                        'fname':    rec['fname'],
                        'path':     rec['path'],
                        'video_id': vid,
                        'frame_no': rec.get('frame_no', -1),
                        'det':      det,
                    })

    return fp_candidates


# ══════════════════════════════════════════════════════════════════════════════
# bbox 불량 후보 (CASE C)
# ══════════════════════════════════════════════════════════════════════════════

def find_bad_bbox(records: List[Dict]) -> List[Dict]:
    """CASE C: bbox 품질 불량."""
    bad = []
    for rec in records:
        for det in rec['detections']:
            if det['class_id'] != 0:
                continue
            quality = classify_bbox_quality(det)
            if quality != 'GOOD':
                bad.append({
                    'case':    'C',
                    'reason':  f"bbox {quality}",
                    'fname':   rec['fname'],
                    'path':    rec['path'],
                    'frame_no': rec.get('frame_no', -1),
                    'det':     det,
                })
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# 저신뢰 탐지 (CASE H: 검수 필요)
# ══════════════════════════════════════════════════════════════════════════════

def find_low_conf(records: List[Dict]) -> List[Dict]:
    """conf 0.10~0.30 구간 : 검수 필요 후보."""
    result = []
    for rec in records:
        for det in rec['detections']:
            if det['class_id'] != 0:
                continue
            if CONF_LOW <= det['conf'] < 0.30:
                result.append({
                    'case':    'H',
                    'reason':  f"저신뢰(conf={det['conf']:.3f})",
                    'fname':   rec['fname'],
                    'path':    rec['path'],
                    'frame_no': rec.get('frame_no', -1),
                    'det':     det,
                })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 저장 (bbox 오버레이 포함)
# ══════════════════════════════════════════════════════════════════════════════

def save_case_image(src_path: str, out_path: str,
                    dets: List[Dict] = None,
                    ref_dets: List[Dict] = None,
                    case: str = '?',
                    reason: str = ''):
    """케이스 이미지 저장 (bbox 오버레이 포함)."""
    frame = cv2.imread(src_path)
    if frame is None:
        return False

    # ROI 영역 하이라이트
    roi_overlay = frame.copy()
    cv2.rectangle(roi_overlay,
                  (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H),
                  (0, 255, 255), 2)
    cv2.addWeighted(roi_overlay, 0.3, frame, 0.7, 0, frame)

    # 탐지 bbox 그리기
    if dets:
        for det in dets:
            x1 = ROI_X + det['x1']
            y1 = ROI_Y + det['y1']
            x2 = ROI_X + det['x2']
            y2 = ROI_Y + det['y2']
            color = (0, 80, 255) if det['class_id'] == 0 else (255, 80, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{det['class_name']} {det['conf']:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # 참조 bbox (연한 색)
    if ref_dets:
        for det in ref_dets:
            x1 = ROI_X + det['x1']
            y1 = ROI_Y + det['y1']
            x2 = ROI_X + det['x2']
            y2 = ROI_Y + det['y2']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 1)
            cv2.putText(frame, "REF", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    # 케이스 라벨
    cv2.putText(frame, f"CASE {case}: {reason}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    # detection_results.json 로드
    if not os.path.exists(args.det):
        print(f"[ERROR] detection_results.json 없음: {args.det}")
        sys.exit(1)

    with open(args.det, encoding='utf-8') as f:
        records = json.load(f)

    # video_id, frame_no 복원 (파일명에서 추출)
    for rec in records:
        fname = rec['fname']
        # VID_A_f001234.jpg → video_id=VID_A, frame_no=1234
        parts = fname.replace('.jpg', '').split('_f')
        if len(parts) == 2:
            rec['video_id'] = parts[0]
            try:
                rec['frame_no'] = int(parts[1])
            except ValueError:
                rec['frame_no'] = -1
        else:
            rec['video_id'] = 'UNK'
            rec['frame_no'] = -1

    print(f"[INFO] 총 프레임: {len(records)}")

    # ── 분류 실행 ────────────────────────────────────────────────────────────
    missed = find_missed_by_context(records)
    fps    = find_false_positives(records)
    bad    = find_bad_bbox(records)
    lowc   = find_low_conf(records)

    print(f"\n[분류 결과]")
    print(f"  CASE B (미탐):       {len(missed)}")
    print(f"  CASE C (bbox 불량):  {len(bad)}")
    print(f"  CASE D (오탐 의심):  {len(fps)}")
    print(f"  CASE H (저신뢰):     {len(lowc)}")

    # ── 케이스 이미지 저장 ───────────────────────────────────────────────────
    missed_dir = os.path.join(args.out, 'missed')
    fp_dir     = os.path.join(args.out, 'false_positives')
    hard_dir   = os.path.join(args.out, 'hard_cases')

    os.makedirs(missed_dir, exist_ok=True)
    os.makedirs(fp_dir,     exist_ok=True)
    os.makedirs(hard_dir,   exist_ok=True)

    # CASE B 저장 (최우선)
    for item in missed[:200]:  # 최대 200장
        out = os.path.join(missed_dir, item['fname'])
        save_case_image(item['path'], out,
                        dets=None,
                        ref_dets=item.get('ref_dets', []),
                        case='B', reason=item['reason'])

    # CASE D 저장
    for item in fps[:100]:
        out = os.path.join(fp_dir, item['fname'])
        save_case_image(item['path'], out,
                        dets=[item['det']],
                        case='D', reason=item['reason'])

    # CASE C + H 저장
    for item in (bad + lowc)[:150]:
        out = os.path.join(hard_dir, item['fname'])
        save_case_image(item['path'], out,
                        dets=[item['det']],
                        case=item['case'], reason=item['reason'])

    # ── JSON 저장 ────────────────────────────────────────────────────────────
    all_cases = {
        'case_B_missed':        missed,
        'case_C_bad_bbox':      bad,
        'case_D_false_positive': fps,
        'case_H_low_conf':      lowc,
        'summary': {
            'total_frames':   len(records),
            'case_B':         len(missed),
            'case_C':         len(bad),
            'case_D':         len(fps),
            'case_H':         len(lowc),
        }
    }

    cases_path = os.path.join(args.out, 'reports', 'hard_cases.json')
    os.makedirs(os.path.dirname(cases_path), exist_ok=True)
    with open(cases_path, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  저장 완료:")
    print(f"    missed/:         {len(missed[:200])}장")
    print(f"    false_positives/: {len(fps[:100])}장")
    print(f"    hard_cases/:     {len((bad+lowc)[:150])}장")
    print(f"    reports/hard_cases.json")

    return all_cases


def main():
    parser = argparse.ArgumentParser(description='미탐/오탐/불량 bbox 분류')
    parser.add_argument('--det',    required=True,
                        help='detection_results.json 경로')
    parser.add_argument('--frames', required=True,
                        help='원본 프레임 디렉토리')
    parser.add_argument('--out',    default='dataset_field_v1',
                        help='출력 루트 (missed/, false_positives/, hard_cases/ 생성)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
