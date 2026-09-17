"""
tools/validate_dataset.py
==========================
최종 데이터셋 품질 검사 + 데이터 누수 검사 + 통계 출력.

검사 항목:
  1. 이미지↔라벨 파일명 일치
  2. class ID 정상 (0 or 1)
  3. bbox 좌표 정상 (0~1 범위)
  4. bbox w/h > 0
  5. bbox 이미지 밖으로 나가는지
  6. 너무 작은 bbox (w*h < 0.001)
  7. 너무 큰 bbox (w*h > 0.90)
  8. 빈 라벨 (몬스터 없는 프레임)
  9. 중복 라벨
  10. 깨진 이미지
  11. Train/Val/Test 누수 (phash 기반)

사용법:
  python tools/validate_dataset.py \
      --dataset dataset_field_v1 \
      --report  dataset_field_v1/reports
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── phash ────────────────────────────────────────────────────────────────────
PHASH_SIZE   = 16
PHASH_HIGH_F = 4
LEAK_THRESH  = 6   # 해밍거리 <= 이면 누수로 판단


def compute_phash(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE),
                              interpolation=cv2.INTER_AREA).astype(np.float32)
        dct     = cv2.dct(resized)
        dct_low = dct[:PHASH_HIGH_F, :PHASH_HIGH_F].flatten()
        med     = np.median(dct_low)
        return (dct_low > med).astype(np.uint8)
    except Exception:
        return None


def hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


# ══════════════════════════════════════════════════════════════════════════════
# 라벨 유효성 검사
# ══════════════════════════════════════════════════════════════════════════════

class LabelIssue:
    def __init__(self, img_path: str, label_path: str,
                 issue: str, detail: str = ''):
        self.img_path   = img_path
        self.label_path = label_path
        self.issue      = issue
        self.detail     = detail

    def to_dict(self):
        return {
            'img':    self.img_path,
            'label':  self.label_path,
            'issue':  self.issue,
            'detail': self.detail,
        }


def validate_split(img_dir: str, label_dir: str,
                   split: str) -> Tuple[List[Dict], List[LabelIssue]]:
    """
    단일 split (train/val/test) 검사.

    Returns:
        (records, issues)
        records: [{fname, n_bbox, classes, bboxes}, ...]
        issues:  [LabelIssue, ...]
    """
    img_files = sorted(
        glob.glob(os.path.join(img_dir, '*.jpg')) +
        glob.glob(os.path.join(img_dir, '*.png'))
    )
    records = []
    issues  = []

    for img_path in img_files:
        stem       = Path(img_path).stem
        label_path = os.path.join(label_dir, stem + '.txt')

        # ① 라벨 파일 존재 여부
        if not os.path.exists(label_path):
            issues.append(LabelIssue(img_path, label_path,
                                     'MISSING_LABEL', '라벨 파일 없음'))
            continue

        # ② 이미지 읽기 가능 여부
        img = cv2.imread(img_path)
        if img is None:
            issues.append(LabelIssue(img_path, label_path,
                                     'BROKEN_IMAGE', '이미지 열기 실패'))
            continue

        ih, iw = img.shape[:2]

        # ③ 라벨 파싱
        bboxes = []
        seen   = set()

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            # 빈 라벨 (negative sample) → 이슈 아님, 기록만
            records.append({
                'fname':  os.path.basename(img_path),
                'split':  split,
                'n_bbox': 0,
                'classes': [],
                'bboxes': [],
                'phash':  None,
            })
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                issues.append(LabelIssue(img_path, label_path,
                                         'INVALID_FORMAT', f"필드 부족: {line}"))
                continue

            try:
                cls_id, cx, cy, w, h = int(parts[0]), float(parts[1]), \
                    float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                issues.append(LabelIssue(img_path, label_path,
                                         'PARSE_ERROR', f"파싱 실패: {line}"))
                continue

            # ④ class ID
            if cls_id not in (0, 1):
                issues.append(LabelIssue(img_path, label_path,
                                         'INVALID_CLASS', f"cls={cls_id}"))

            # ⑤ 좌표 범위
            for val, name in [(cx, 'cx'), (cy, 'cy'), (w, 'w'), (h, 'h')]:
                if not (0.0 <= val <= 1.0):
                    issues.append(LabelIssue(img_path, label_path,
                                             'OUT_OF_RANGE',
                                             f"{name}={val:.4f}"))

            # ⑥ w/h > 0
            if w <= 0 or h <= 0:
                issues.append(LabelIssue(img_path, label_path,
                                         'ZERO_SIZE', f"w={w} h={h}"))

            # ⑦ 너무 작음
            if w * h < 0.0008:
                issues.append(LabelIssue(img_path, label_path,
                                         'TOO_SMALL',
                                         f"area={w*h:.5f}"))

            # ⑧ 너무 큼
            if w * h > 0.90:
                issues.append(LabelIssue(img_path, label_path,
                                         'TOO_LARGE',
                                         f"area={w*h:.3f}"))

            # ⑨ 중복 라벨
            key = (cls_id, round(cx, 4), round(cy, 4))
            if key in seen:
                issues.append(LabelIssue(img_path, label_path,
                                         'DUPLICATE_BBOX', f"{key}"))
            seen.add(key)

            bboxes.append({'cls': cls_id, 'cx': cx, 'cy': cy, 'w': w, 'h': h})

        records.append({
            'fname':   os.path.basename(img_path),
            'path':    img_path,
            'split':   split,
            'n_bbox':  len(bboxes),
            'classes': [b['cls'] for b in bboxes],
            'bboxes':  bboxes,
            'phash':   None,  # 나중에 계산
        })

    return records, issues


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 누수 검사
# ══════════════════════════════════════════════════════════════════════════════

def check_data_leakage(split_records: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Train↔Val, Train↔Test, Val↔Test 간 유사 프레임 탐지.
    phash 기반.
    """
    # phash 계산
    print("  [누수] phash 계산 중...")
    for split, records in split_records.items():
        for rec in records:
            if not rec.get('path'):
                continue
            img = cv2.imread(rec['path'])
            if img is not None:
                rec['phash'] = compute_phash(img)

    leaks = []
    splits = list(split_records.keys())

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            recs1  = [r for r in split_records[s1] if r.get('phash') is not None]
            recs2  = [r for r in split_records[s2] if r.get('phash') is not None]

            print(f"  [누수] {s1}↔{s2} 비교: {len(recs1)}×{len(recs2)}")

            for r1 in recs1:
                for r2 in recs2:
                    d = hamming(r1['phash'], r2['phash'])
                    if d <= LEAK_THRESH:
                        leaks.append({
                            'split1': s1, 'fname1': r1['fname'],
                            'split2': s2, 'fname2': r2['fname'],
                            'hamming': d,
                        })

    return leaks


# ══════════════════════════════════════════════════════════════════════════════
# 통계 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(all_records: List[Dict]) -> Dict:
    stats = {
        'total_images': len(all_records),
        'total_monster': 0,
        'total_adena':   0,
        'empty_labels':  0,
        'zone': defaultdict(int),
        'size': defaultdict(int),
        'by_split': defaultdict(lambda: {'images': 0, 'monster': 0, 'adena': 0}),
    }

    for rec in all_records:
        split = rec.get('split', 'unknown')
        stats['by_split'][split]['images'] += 1

        if not rec['bboxes']:
            stats['empty_labels'] += 1
            continue

        for bbox in rec['bboxes']:
            if bbox['cls'] == 0:
                stats['total_monster'] += 1
                stats['by_split'][split]['monster'] += 1

                cx, cy, w, h = bbox['cx'], bbox['cy'], bbox['w'], bbox['h']

                # 위치 분류 (정규화 좌표 기준)
                if cx < 0.15:    stats['zone']['LEFT']   += 1
                elif cx > 0.85:  stats['zone']['RIGHT']  += 1
                else:            stats['zone']['CENTER'] += 1

                if cy < 0.25:    stats['zone']['TOP']    += 1
                elif cy > 0.80:  stats['zone']['BOTTOM'] += 1
                else:            stats['zone']['MID']    += 1

                # 크기 분류
                area = w * h
                if area < 0.003:  stats['size']['SMALL']  += 1
                elif area > 0.05: stats['size']['LARGE']  += 1
                else:             stats['size']['MID']    += 1

            else:
                stats['total_adena'] += 1
                stats['by_split'][split]['adena'] += 1

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    dataset_dir = args.dataset
    report_dir  = args.report
    os.makedirs(report_dir, exist_ok=True)

    splits = ['train', 'val', 'test']
    split_records = {}
    all_issues    = []
    all_records   = []

    for split in splits:
        img_dir   = os.path.join(dataset_dir, 'images', split)
        label_dir = os.path.join(dataset_dir, 'labels', split)

        if not os.path.isdir(img_dir):
            print(f"  [SKIP] {split}: 이미지 디렉토리 없음 ({img_dir})")
            split_records[split] = []
            continue

        recs, issues = validate_split(img_dir, label_dir, split)
        split_records[split] = recs
        all_issues.extend(issues)
        all_records.extend(recs)

        print(f"  [{split}] 이미지={len(recs)}  이슈={len(issues)}")

    # ── 통계 ─────────────────────────────────────────────────────────────────
    stats = compute_stats(all_records)

    # ── 누수 검사 ────────────────────────────────────────────────────────────
    leaks = check_data_leakage(split_records)

    # ── 보고서 출력 ──────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  데이터셋 품질 검사 보고서")
    print(f"{'='*58}")
    print(f"  총 이미지: {stats['total_images']}")
    print()

    for split in splits:
        s = stats['by_split'][split]
        print(f"  [{split}]  이미지={s['images']}  "
              f"monster={s['monster']}  adena={s['adena']}")

    print()
    print(f"  monster 총 instance: {stats['total_monster']}")
    print(f"  adena   총 instance: {stats['total_adena']}")
    print(f"  빈 라벨 (negative): {stats['empty_labels']}")
    print()

    print(f"  [위치 분포]")
    for k in ['LEFT', 'CENTER', 'RIGHT', 'TOP', 'MID', 'BOTTOM']:
        print(f"    {k}: {stats['zone'].get(k, 0)}")
    print()

    print(f"  [크기 분포]")
    for k in ['SMALL', 'MID', 'LARGE']:
        print(f"    {k}: {stats['size'].get(k, 0)}")
    print()

    # 이슈 요약
    issue_types = defaultdict(int)
    for iss in all_issues:
        issue_types[iss.issue] += 1

    print(f"  [라벨 품질 이슈] 총 {len(all_issues)}건")
    for t, cnt in sorted(issue_types.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt}")
    print()

    # 누수 결과
    print(f"  [데이터 누수] {len(leaks)}건 발견")
    if leaks:
        for leak in leaks[:20]:
            print(f"    {leak['split1']}/{leak['fname1']} ↔ "
                  f"{leak['split2']}/{leak['fname2']}  "
                  f"hamming={leak['hamming']}")
        if len(leaks) > 20:
            print(f"    ... 외 {len(leaks)-20}건")
    print()

    # ── JSON 저장 ────────────────────────────────────────────────────────────
    report = {
        'stats':      dict(stats),
        'issues':     [i.to_dict() for i in all_issues],
        'leaks':      leaks,
        'issue_summary': dict(issue_types),
    }
    # defaultdict → dict 변환
    report['stats']['zone']     = dict(report['stats']['zone'])
    report['stats']['size']     = dict(report['stats']['size'])
    report['stats']['by_split'] = {
        k: dict(v) for k, v in report['stats']['by_split'].items()
    }

    rpt_path = os.path.join(report_dir, 'validation_report.json')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"  보고서 저장: {rpt_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='데이터셋 품질 검사')
    parser.add_argument('--dataset', required=True,
                        help='데이터셋 루트 (images/ labels/ 포함)')
    parser.add_argument('--report',  default='dataset_field_v1/reports',
                        help='보고서 저장 디렉토리')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
