"""
tools/generate_yolo_labels.py
==============================
추출된 프레임에 기존 YOLO 모델(v1-2)로 1차 자동 라벨링을 수행한다.

출력:
  - 각 프레임에 대한 YOLO format .txt 라벨
  - detection_results.json : 프레임별 상세 탐지 결과
  - label_summary.json     : 통계 요약

클래스:
  0 = monster
  1 = adena

사용법:
  python tools/generate_yolo_labels.py \
      --frames dataset_field_v1/frames \
      --out    dataset_field_v1/labels_auto \
      --model  runs/detect/monster_v1-2/weights/best.pt \
      --conf   0.10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── ROI 설정 ─────────────────────────────────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472

# ── 클래스 ───────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: 'monster', 1: 'adena'}

# ── confidence 구간 정의 ─────────────────────────────────────────────────────
CONF_HIGH   = 0.50   # 고신뢰 (GOOD 후보)
CONF_MID    = 0.20   # 중간 (검수 필요)
CONF_LOW    = 0.10   # 저신뢰 (오탐 의심)


# ══════════════════════════════════════════════════════════════════════════════
# YOLO 추론
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, device: str = 'cpu'):
    """YOLO 모델 로드."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    # 워밍업
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, imgsz=640, verbose=False, device=device)
    print(f"[MODEL] 로드: {model_path}  device={device}")
    return model


def infer_roi(model, frame_bgr: np.ndarray,
              conf: float = 0.10, iou: float = 0.45,
              imgsz: int = 640, device: str = 'cpu') -> List[Dict]:
    """
    전체 프레임에서 ROI 크롭 후 YOLO 추론.
    결과 bbox는 전체 이미지 좌표(절대값) 및 정규화 좌표 모두 반환.
    """
    roi = frame_bgr[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    results = model(
        roi,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])
            cls_id   = int(box.cls[0])

            if cls_id not in CLASS_NAMES:
                continue

            # ROI 내부 절대 좌표
            w_abs = x2 - x1
            h_abs = y2 - y1
            cx_roi = (x1 + x2) / 2
            cy_roi = (y1 + y2) / 2

            # YOLO 정규화 좌표 (ROI 기준)
            cx_n  = cx_roi / ROI_W
            cy_n  = cy_roi / ROI_H
            w_n   = w_abs  / ROI_W
            h_n   = h_abs  / ROI_H

            # 클리핑 확인
            clipped = (x1 <= 0 or y1 <= 0 or
                       x2 >= ROI_W - 1 or y2 >= ROI_H - 1)

            # 위치 영역 분류
            zone_h = ('LEFT'   if cx_roi < ROI_W * 0.15 else
                      'RIGHT'  if cx_roi > ROI_W * 0.85 else 'CENTER')
            zone_v = ('TOP'    if cy_roi < ROI_H * 0.20 else
                      'BOTTOM' if cy_roi > ROI_H * 0.80 else 'MID')

            # 크기 분류
            area = w_abs * h_abs
            size = ('SMALL' if area < 1600 else
                    'LARGE' if w_abs > 100 or h_abs > 100 else 'MID')

            detections.append({
                'class_id':   cls_id,
                'class_name': CLASS_NAMES[cls_id],
                'conf':       round(conf_val, 4),
                # ROI 내 절대 좌표
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx_roi': round(cx_roi, 1),
                'cy_roi': round(cy_roi, 1),
                'w_abs':  w_abs,
                'h_abs':  h_abs,
                # YOLO 정규화 (ROI 기준)
                'cx_n': round(cx_n, 6),
                'cy_n': round(cy_n, 6),
                'w_n':  round(w_n,  6),
                'h_n':  round(h_n,  6),
                # 메타
                'clipped': clipped,
                'zone_h':  zone_h,
                'zone_v':  zone_v,
                'size':    size,
            })

    return detections


# ══════════════════════════════════════════════════════════════════════════════
# confidence 기반 탐지 품질 판단
# ══════════════════════════════════════════════════════════════════════════════

def classify_det(det: Dict) -> str:
    """
    단일 탐지의 품질 분류.
    GOOD   : conf >= 0.50, 클리핑 없음
    MID    : 0.20 <= conf < 0.50
    LOW    : 0.10 <= conf < 0.20  (저신뢰, 오탐 가능)
    """
    c = det['conf']
    if det['clipped']:
        return 'CLIPPED'
    if c >= CONF_HIGH:
        return 'GOOD'
    if c >= CONF_MID:
        return 'MID'
    return 'LOW'


# ══════════════════════════════════════════════════════════════════════════════
# YOLO label 파일 생성
# ══════════════════════════════════════════════════════════════════════════════

def write_yolo_label(label_path: str, detections: List[Dict],
                     min_conf: float = 0.10):
    """YOLO format .txt 라벨 파일 저장."""
    lines = []
    for det in detections:
        if det['conf'] < min_conf:
            continue
        # class cx_n cy_n w_n h_n  (ROI 기준 정규화)
        lines.append(
            f"{det['class_id']} "
            f"{det['cx_n']:.6f} {det['cy_n']:.6f} "
            f"{det['w_n']:.6f} {det['h_n']:.6f}"
        )
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))
    return len(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    # 모델 로드
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 없음: {model_path}")
        sys.exit(1)

    model = load_model(model_path, device=args.device)

    # 프레임 목록
    frame_files = sorted(
        glob.glob(os.path.join(args.frames, '*.jpg')) +
        glob.glob(os.path.join(args.frames, '*.png'))
    )
    if not frame_files:
        print(f"[ERROR] 프레임 없음: {args.frames}")
        sys.exit(1)

    print(f"[INFO] 처리할 프레임: {len(frame_files)}장")

    os.makedirs(args.out, exist_ok=True)

    # ── 추론 루프 ────────────────────────────────────────────────────────────
    all_results = []
    stats = {
        'total':           len(frame_files),
        'det_any':         0,
        'det_monster':     0,
        'det_adena':       0,
        'no_det':          0,
        'conf_high':       0,
        'conf_mid':        0,
        'conf_low':        0,
        'clipped':         0,
        'zone': {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0},
        'vert': {'TOP': 0, 'MID': 0, 'BOTTOM': 0},
        'size': {'SMALL': 0, 'MID': 0, 'LARGE': 0},
    }

    for idx, fpath in enumerate(frame_files):
        fname = os.path.basename(fpath)
        stem  = Path(fpath).stem

        frame = cv2.imread(fpath)
        if frame is None:
            print(f"  [SKIP] 이미지 읽기 실패: {fpath}")
            continue

        dets = infer_roi(model, frame,
                         conf=args.conf, iou=args.iou,
                         imgsz=args.imgsz, device=args.device)

        # 라벨 파일 저장
        label_path = os.path.join(args.out, stem + '.txt')
        n_written  = write_yolo_label(label_path, dets, min_conf=args.conf)

        # 통계
        if dets:
            stats['det_any'] += 1
        else:
            stats['no_det'] += 1

        frame_rec = {
            'fname':       fname,
            'path':        fpath,
            'label_path':  label_path,
            'n_det':       len(dets),
            'detections':  dets,
        }

        for det in dets:
            if det['class_id'] == 0:
                stats['det_monster'] += 1
            else:
                stats['det_adena'] += 1

            q = classify_det(det)
            if q == 'GOOD':     stats['conf_high'] += 1
            elif q == 'MID':    stats['conf_mid']  += 1
            elif q == 'LOW':    stats['conf_low']  += 1
            elif q == 'CLIPPED': stats['clipped']  += 1

            if det['class_id'] == 0:
                stats['zone'][det['zone_h']] += 1
                stats['vert'][det['zone_v']] += 1
                stats['size'][det['size']]   += 1

        all_results.append(frame_rec)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(frame_files)}] "
                  f"탐지: {stats['det_any']}  미탐: {stats['no_det']}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    det_path = os.path.join(args.out, 'detection_results.json')
    with open(det_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    sum_path = os.path.join(args.out, 'label_summary.json')
    with open(sum_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── 출력 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  자동 라벨링 완료")
    print(f"{'='*60}")
    print(f"  총 프레임:       {stats['total']}")
    print(f"  탐지 (≥conf):    {stats['det_any']} "
          f"({stats['det_any']/stats['total']*100:.1f}%)")
    print(f"  미탐:            {stats['no_det']} "
          f"({stats['no_det']/stats['total']*100:.1f}%)")
    print()
    print(f"  [monster 탐지]")
    print(f"    총 instance:   {stats['det_monster']}")
    print(f"    GOOD(≥0.50):  {stats['conf_high']}")
    print(f"    MID(0.20~):   {stats['conf_mid']}")
    print(f"    LOW(0.10~):   {stats['conf_low']}")
    print(f"    CLIPPED:      {stats['clipped']}")
    print()
    print(f"  [위치 분포]")
    for k, v in stats['zone'].items():
        print(f"    {k}: {v}")
    for k, v in stats['vert'].items():
        print(f"    {k}: {v}")
    print(f"  [크기 분포]")
    for k, v in stats['size'].items():
        print(f"    {k}: {v}")
    print()
    print(f"  결과: {det_path}")
    print(f"  통계: {sum_path}")

    return all_results, stats


def main():
    parser = argparse.ArgumentParser(description='YOLO 1차 자동 라벨링')
    parser.add_argument('--frames', required=True,
                        help='추출 프레임 디렉토리')
    parser.add_argument('--out',    required=True,
                        help='라벨 출력 디렉토리')
    parser.add_argument('--model',
                        default='runs/detect/monster_v1-2/weights/best.pt',
                        help='YOLO 모델 경로')
    parser.add_argument('--conf',   type=float, default=0.10,
                        help='추론 confidence 임계값 (기본: 0.10)')
    parser.add_argument('--iou',    type=float, default=0.45)
    parser.add_argument('--imgsz',  type=int,   default=640)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
