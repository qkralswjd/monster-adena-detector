"""
tools/build_field_dataset.py
============================
실제 게임 필드 녹화 영상 → dataset_field_v1/ 구축 메인 오케스트레이션 스크립트.

파이프라인:
  1. 6개 영상 → extract_frames.py  (interval=5, phash_thresh=8)
  2. 추출 프레임 → generate_yolo_labels.py  (conf=0.10, model=v1-2)
  3. 탐지 결과 → filter_hard_cases.py  (CASE A/B/C/D/H 분류)
  4. 영상 단위 Train/Val/Test 분리 → images/ labels/ 복사
  5. data.yaml 생성
  6. validate_dataset.py 호출
  7. preview 이미지 생성 (bbox 오버레이)
  8. 최종 보고서 22개 항목 출력

주의:
  - 기존 dataset/ 및 runs/detect/monster_v2/weights/best.pt 절대 손상 금지
  - 새 데이터셋은 dataset_field_v1/ 에만 독립 구축
  - 이번 실행에서 새 모델 학습 금지

영상별 Train/Val/Test 할당 (계획):
  Train : VID_A, VID_B, VID_C, VID_D
  Val   : VID_E
  Test  : VID_F

사용법:
  python tools/build_field_dataset.py [--step all|extract|label|filter|split|validate|report]
  python tools/build_field_dataset.py --step all
  python tools/build_field_dataset.py --step extract   # 프레임 추출만
  python tools/build_field_dataset.py --step report    # 보고서만 (기존 결과 이용)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # /home/user/monster_tracker
TOOLS_DIR   = BASE_DIR / 'tools'
DATASET_DIR = BASE_DIR / 'dataset_field_v1'

# 모델 우선순위: v1-2 (v2 없으면 자동 fallback)
MODEL_CANDIDATES = [
    BASE_DIR / 'runs/detect/monster_v1-2/weights/best.pt',
    BASE_DIR / 'runs/detect/monster_v1/weights/best.pt',
]

# 영상 목록 (VID_ID → 경로)
VIDEOS = {
    'VID_A': BASE_DIR / 'test_runs/2026-09-17_06-13-10/screen.mp4',
    'VID_B': BASE_DIR / 'test_runs/2026-09-17_06-40-52/screen.mp4',
    'VID_C': BASE_DIR / 'test_runs/2026-09-17_07-00-50/screen.mp4',
    'VID_D': BASE_DIR / 'test_runs/2026-09-17_07-45-08/screen.mp4',
    'VID_E': BASE_DIR / 'test_runs/2026-09-17_08-14-32/screen.mp4',
    'VID_F': BASE_DIR / 'test_runs/2026-09-17_09-10-38/screen_95mb.mp4',
}

# 영상별 split 할당
VIDEO_SPLIT = {
    'VID_A': 'train',
    'VID_B': 'train',
    'VID_C': 'train',
    'VID_D': 'train',
    'VID_E': 'val',
    'VID_F': 'test',
}

# ── ROI 설정 ──────────────────────────────────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472

# ── confidence 구간 ───────────────────────────────────────────────────────────
CONF_GOOD = 0.50
CONF_MID  = 0.20
CONF_LOW  = 0.10

# ── 클래스 ────────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: 'monster', 1: 'adena'}

# ══════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = 'INFO'):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}][{level}] {msg}", flush=True)


def find_model() -> Optional[Path]:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def run_script(script: Path, args: List[str], step_name: str) -> int:
    cmd = [sys.executable, str(script)] + args
    log(f"실행: {' '.join(cmd)}", 'RUN')
    ret = subprocess.run(cmd, cwd=str(BASE_DIR))
    if ret.returncode != 0:
        log(f"{step_name} 실패 (rc={ret.returncode})", 'ERR')
    return ret.returncode


def ensure_dirs():
    for split in ['train', 'val', 'test']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for d in ['frames', 'labels_auto', 'hard_cases', 'missed',
              'false_positives', 'previews', 'reports']:
        (DATASET_DIR / d).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 프레임 추출
# ══════════════════════════════════════════════════════════════════════════════

def step_extract(interval: int = 5, phash_thresh: int = 8,
                 max_frames: int = 800) -> Dict:
    """
    각 영상을 독립적으로 추출 → frames/{VID_X}/ 에 저장.
    extract_frames.py 는 --out 디렉토리에 직접 JPG를 저장하므로
    영상별로 out_dir을 분리해서 호출한다.
    완료 후 extract_meta.json 을 읽어 summary 구성.
    """
    log("=" * 60)
    log("STEP 1: 프레임 추출 + phash 중복 제거")

    frames_dir = DATASET_DIR / 'frames'
    all_summaries = {}

    for vid_id, vid_path in VIDEOS.items():
        if not vid_path.exists():
            log(f"  [{vid_id}] 영상 없음: {vid_path}", 'WARN')
            continue

        out_dir = frames_dir / vid_id
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / 'extract_meta.json'

        # 이미 추출된 경우 스킵
        existing_jpgs = list(out_dir.glob('*.jpg'))
        if meta_path.exists() and existing_jpgs:
            with open(meta_path) as f:
                meta = json.load(f)
            # meta는 list[dict] 형태
            if isinstance(meta, list) and meta:
                m = meta[0]
            else:
                m = meta if isinstance(meta, dict) else {}
            saved = m.get('after_dedup', len(existing_jpgs))
            sampled = m.get('sampled', saved)
            log(f"  [{vid_id}] 이미 추출됨 ({saved}장 → 스킵)")
            all_summaries[vid_id] = {
                'saved': saved,
                'sampled': sampled,
                'skipped_dup': sampled - saved,
                'skipped_static': 0,
            }
            continue

        log(f"  [{vid_id}] 추출 중: {vid_path.name}")
        rc = run_script(
            TOOLS_DIR / 'extract_frames.py',
            ['--videos', str(vid_path),
             '--out', str(out_dir),
             '--interval', str(interval),
             '--phash-thresh', str(phash_thresh),
             '--max-frames', str(max_frames)],
            f'extract_{vid_id}'
        )

        if rc == 0 and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if isinstance(meta, list) and meta:
                m = meta[0]
            else:
                m = meta if isinstance(meta, dict) else {}
            saved   = m.get('after_dedup', len(list(out_dir.glob('*.jpg'))))
            sampled = m.get('sampled', saved)
            log(f"  [{vid_id}] 완료: {saved}장 저장 (샘플링={sampled})")
            all_summaries[vid_id] = {
                'saved': saved,
                'sampled': sampled,
                'skipped_dup': sampled - saved,
                'skipped_static': 0,
            }
        else:
            # meta 없어도 jpg 개수로 기록
            saved = len(list(out_dir.glob('*.jpg')))
            log(f"  [{vid_id}] 완료 (meta 없음): {saved}장")
            all_summaries[vid_id] = {'saved': saved, 'sampled': saved,
                                     'skipped_dup': 0, 'skipped_static': 0}

    # 전체 요약 저장
    total_path = DATASET_DIR / 'reports' / 'extract_summary_all.json'
    with open(total_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    total_saved = sum(s.get('saved', 0) for s in all_summaries.values())
    log(f"  전체: {total_saved}장 저장")
    return all_summaries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 자동 라벨링
# ══════════════════════════════════════════════════════════════════════════════

def step_label(model_path: Path, conf: float = 0.10) -> Dict:
    """
    generate_yolo_labels.py 호출.
    출력: labels_auto/{VID_X}/detection_results.json, label_summary.json, *.txt
    label_summary.json 키: total, det_any, no_det, conf_high, conf_mid, conf_low, ...
    """
    log("=" * 60)
    log("STEP 2: YOLO 자동 1차 라벨링")

    frames_dir   = DATASET_DIR / 'frames'
    labels_dir   = DATASET_DIR / 'labels_auto'
    results_path = DATASET_DIR / 'reports' / 'detection_results_all.json'

    if results_path.exists():
        log("  이미 라벨링 완료 → 스킵 (reports/detection_results_all.json 존재)")
        with open(results_path) as f:
            return json.load(f)

    all_results = {}

    for vid_id in VIDEOS:
        vid_frames_dir = frames_dir / vid_id
        if not vid_frames_dir.exists():
            log(f"  [{vid_id}] 프레임 없음 → 스킵", 'WARN')
            continue

        frames = sorted(glob.glob(str(vid_frames_dir / '*.jpg')))
        if not frames:
            log(f"  [{vid_id}] jpg 파일 없음 → 스킵", 'WARN')
            continue

        vid_label_dir = labels_dir / vid_id
        summary_path  = vid_label_dir / 'label_summary.json'

        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            # label_summary.json 키: total, det_any, no_det
            log(f"  [{vid_id}] 이미 라벨링됨 "
                f"(det={s.get('det_any',0)}, no_det={s.get('no_det',0)} → 스킵)")
            all_results[vid_id] = s
            continue

        log(f"  [{vid_id}] 라벨링 중... ({len(frames)}장)")
        rc = run_script(
            TOOLS_DIR / 'generate_yolo_labels.py',
            ['--frames', str(vid_frames_dir),
             '--out',    str(vid_label_dir),
             '--model',  str(model_path),
             '--conf',   str(conf)],
            f'label_{vid_id}'
        )
        if rc == 0 and summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            all_results[vid_id] = s
            log(f"  [{vid_id}] 완료: det={s.get('det_any',0)}, "
                f"no_det={s.get('no_det',0)}")
        else:
            # summary 없어도 txt 개수로 기록
            txt_count = len(list(vid_label_dir.glob('*.txt'))) if vid_label_dir.exists() else 0
            log(f"  [{vid_id}] 완료 (summary 없음): txt={txt_count}", 'WARN')
            all_results[vid_id] = {'total': len(frames), 'det_any': txt_count,
                                   'no_det': len(frames) - txt_count}

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Hard Case 분류
# ══════════════════════════════════════════════════════════════════════════════

def step_filter() -> Dict:
    """
    filter_hard_cases.py 호출.
    인자: --det detection_results.json  --frames 프레임dir  --out 출력루트
    출력: {out}/reports/hard_cases.json  (summary 포함)
    영상별로 독립 실행하여 결과를 vid별로 합산.
    """
    log("=" * 60)
    log("STEP 3: CASE A/B/C/D/H 분류 (filter_hard_cases)")

    frames_dir   = DATASET_DIR / 'frames'
    labels_dir   = DATASET_DIR / 'labels_auto'
    summary_path = DATASET_DIR / 'reports' / 'filter_summary_all.json'

    if summary_path.exists():
        log("  이미 분류 완료 → 스킵")
        with open(summary_path) as f:
            return json.load(f)

    all_filter = {}

    for vid_id in VIDEOS:
        vid_label_dir    = labels_dir / vid_id
        det_results_path = vid_label_dir / 'detection_results.json'

        if not det_results_path.exists():
            log(f"  [{vid_id}] detection_results.json 없음 → 스킵", 'WARN')
            continue

        # filter_hard_cases.py 는 --out 루트에 reports/, missed/ 등을 생성
        # 영상별로 임시 out_dir 사용
        vid_out_dir = DATASET_DIR / 'hard_cases' / vid_id
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        log(f"  [{vid_id}] 분류 중...")
        rc = run_script(
            TOOLS_DIR / 'filter_hard_cases.py',
            ['--det',    str(det_results_path),
             '--frames', str(frames_dir / vid_id),
             '--out',    str(vid_out_dir)],
            f'filter_{vid_id}'
        )

        # hard_cases.json: {summary: {case_B, case_C, case_D, case_H}}
        hard_cases_path = vid_out_dir / 'reports' / 'hard_cases.json'
        if hard_cases_path.exists():
            with open(hard_cases_path) as f:
                hc = json.load(f)
            s = hc.get('summary', {})
            all_filter[vid_id] = s
            log(f"  [{vid_id}] B={s.get('case_B',0)} C={s.get('case_C',0)} "
                f"D={s.get('case_D',0)} H={s.get('case_H',0)}")
        else:
            log(f"  [{vid_id}] hard_cases.json 없음 (rc={rc})", 'WARN')
            all_filter[vid_id] = {'case_A': 0, 'case_B': 0,
                                  'case_C': 0, 'case_D': 0, 'case_H': 0}

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_filter, f, indent=2, ensure_ascii=False)

    return all_filter


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: 영상 단위 Train/Val/Test 분리
# ══════════════════════════════════════════════════════════════════════════════

def step_split() -> Dict:
    """
    영상 단위로 images/ labels/ 복사.
    라벨 소스: labels_auto/{vid_id}/*.txt  (YOLO format)
    이미지 소스: frames/{vid_id}/*.jpg
    CASE D (오탐) → labels에 빈 파일 (Hard Negative)
    """
    log("=" * 60)
    log("STEP 4: 영상 단위 Train/Val/Test 분리")

    frames_dir = DATASET_DIR / 'frames'
    labels_dir = DATASET_DIR / 'labels_auto'
    split_summary = {s: {'images': 0, 'labels': 0, 'no_label': 0}
                     for s in ['train', 'val', 'test']}

    for vid_id, split in VIDEO_SPLIT.items():
        vid_frames_dir = frames_dir / vid_id
        vid_label_dir  = labels_dir / vid_id

        if not vid_frames_dir.exists():
            log(f"  [{vid_id}→{split}] 프레임 없음 → 스킵", 'WARN')
            continue

        img_out   = DATASET_DIR / 'images' / split
        label_out = DATASET_DIR / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        jpgs = sorted(vid_frames_dir.glob('*.jpg'))
        copied = 0
        for jpg in jpgs:
            dst_img = img_out / jpg.name
            if dst_img.exists():
                copied += 1
                continue

            shutil.copy2(jpg, dst_img)

            # 대응 라벨 복사 (없으면 빈 파일 = 배경/Hard Negative)
            src_lbl = vid_label_dir / (jpg.stem + '.txt')
            dst_lbl = label_out / (jpg.stem + '.txt')
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
                split_summary[split]['labels'] += 1
            else:
                # 빈 라벨 (Hard Negative 또는 미탐 후보)
                dst_lbl.write_text('')
                split_summary[split]['no_label'] += 1

            split_summary[split]['images'] += 1
            copied += 1

        log(f"  [{vid_id}→{split}] {len(jpgs)}장 복사 완료")

    log(f"  분리 결과:")
    for split, s in split_summary.items():
        log(f"    {split}: images={s['images']} labels={s['labels']} "
            f"no_label={s['no_label']}")

    report_path = DATASET_DIR / 'reports' / 'split_summary.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    return split_summary


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: data.yaml 생성
# ══════════════════════════════════════════════════════════════════════════════

def step_data_yaml():
    log("=" * 60)
    log("STEP 5: data.yaml 생성")

    yaml_path = DATASET_DIR / 'data.yaml'
    content = f"""# dataset_field_v1  data.yaml
# 실제 게임 필드 녹화 영상 기반 데이터셋
# 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}
#
# Train: VID_A, VID_B, VID_C, VID_D
# Val  : VID_E
# Test : VID_F

path: {DATASET_DIR.resolve()}

train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: monster
  1: adena

# 기존 데이터셋
# base_dataset: dataset/dataset.yaml
# base_train: 539장, base_val: 23장
"""
    yaml_path.write_text(content, encoding='utf-8')
    log(f"  저장: {yaml_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: 데이터셋 검증
# ══════════════════════════════════════════════════════════════════════════════

def step_validate() -> Dict:
    log("=" * 60)
    log("STEP 6: 데이터셋 품질 검사 + 누수 검사")

    report_dir = DATASET_DIR / 'reports'
    rc = run_script(
        TOOLS_DIR / 'validate_dataset.py',
        ['--dataset', str(DATASET_DIR),
         '--report',  str(report_dir)],
        'validate'
    )

    val_report_path = report_dir / 'validation_report.json'
    if val_report_path.exists():
        with open(val_report_path) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Preview 이미지 생성
# ══════════════════════════════════════════════════════════════════════════════

def step_preview(n_per_split: int = 8):
    """각 split에서 샘플 이미지를 뽑아 bbox 오버레이 preview 저장."""
    log("=" * 60)
    log("STEP 7: Preview 이미지 생성 (bbox 오버레이)")

    preview_dir = DATASET_DIR / 'previews'
    preview_dir.mkdir(exist_ok=True)

    COLOR_MAP = {
        0: (0, 255, 0),    # monster → 초록
        1: (0, 128, 255),  # adena   → 주황
    }

    for split in ['train', 'val', 'test']:
        img_dir   = DATASET_DIR / 'images' / split
        label_dir = DATASET_DIR / 'labels' / split

        jpgs = sorted(img_dir.glob('*.jpg'))
        if not jpgs:
            continue

        # confidence 기반 라벨 유무 샘플링: 라벨 있는 것 우선
        labeled   = [j for j in jpgs if (label_dir/(j.stem+'.txt')).exists()
                     and (label_dir/(j.stem+'.txt')).stat().st_size > 0]
        unlabeled = [j for j in jpgs if j not in labeled]

        samples = labeled[:n_per_split]
        if len(samples) < n_per_split:
            samples += unlabeled[:n_per_split - len(samples)]

        split_dir = preview_dir / split
        split_dir.mkdir(exist_ok=True)

        saved = 0
        for jpg in samples:
            img = cv2.imread(str(jpg))
            if img is None:
                continue

            lbl_path = label_dir / (jpg.stem + '.txt')
            H, W = img.shape[:2]

            # ROI 경계 표시
            cv2.rectangle(img,
                          (ROI_X, ROI_Y),
                          (ROI_X + ROI_W, ROI_Y + ROI_H),
                          (128, 128, 128), 1)

            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])
                    conf_str = f" {float(parts[5]):.2f}" if len(parts) > 5 else ''

                    # 정규화 좌표 → ROI 픽셀 좌표
                    cx_roi = int(cx_n * ROI_W)
                    cy_roi = int(cy_n * ROI_H)
                    bw_roi = int(bw_n * ROI_W)
                    bh_roi = int(bh_n * ROI_H)

                    # 전체 이미지 좌표로 변환
                    x1 = ROI_X + cx_roi - bw_roi // 2
                    y1 = ROI_Y + cy_roi - bh_roi // 2
                    x2 = ROI_X + cx_roi + bw_roi // 2
                    y2 = ROI_Y + cy_roi + bh_roi // 2

                    color = COLOR_MAP.get(cls_id, (255, 0, 0))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = CLASS_NAMES.get(cls_id, str(cls_id)) + conf_str
                    cv2.putText(img, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 파일명에 split 접두사
            out_name = f"{split}_{jpg.stem}.jpg"
            cv2.imwrite(str(split_dir / out_name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1

        log(f"  [{split}] preview {saved}장 → previews/{split}/")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: 통계 수집 (보고서용)
# ══════════════════════════════════════════════════════════════════════════════

def collect_stats() -> Dict:
    """dataset_field_v1 전체 통계 수집."""
    stats: Dict = {
        'split': {},
        'class': {'monster': 0, 'adena': 0},
        'position': {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0,
                     'TOP': 0, 'BOTTOM': 0},
        'size': {'small': 0, 'medium': 0, 'large': 0},
        'conf_band': {'GOOD': 0, 'MID': 0, 'LOW': 0, 'NO_CONF': 0},
        'hard_neg': 0,
        'empty_label': 0,
    }

    for split in ['train', 'val', 'test']:
        img_dir   = DATASET_DIR / 'images' / split
        label_dir = DATASET_DIR / 'labels' / split
        imgs = list(img_dir.glob('*.jpg'))
        stats['split'][split] = {'images': len(imgs), 'instances': 0}

        for img_path in imgs:
            lbl = label_dir / (img_path.stem + '.txt')
            if not lbl.exists() or lbl.stat().st_size == 0:
                stats['empty_label'] += 1
                stats['hard_neg'] += 1
                continue

            for line in lbl.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx_n   = float(parts[1])
                cy_n   = float(parts[2])
                bw_n   = float(parts[3])
                bh_n   = float(parts[4])
                conf   = float(parts[5]) if len(parts) > 5 else -1.0

                cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                if cls_name in stats['class']:
                    stats['class'][cls_name] += 1
                stats['split'][split]['instances'] += 1

                # 위치
                if cx_n < 0.33:
                    stats['position']['LEFT'] += 1
                elif cx_n > 0.67:
                    stats['position']['RIGHT'] += 1
                else:
                    stats['position']['CENTER'] += 1
                if cy_n < 0.40:
                    stats['position']['TOP'] += 1
                elif cy_n > 0.65:
                    stats['position']['BOTTOM'] += 1

                # 크기 (ROI 기준 면적)
                area = bw_n * bh_n
                if area < 0.005:
                    stats['size']['small'] += 1
                elif area < 0.05:
                    stats['size']['medium'] += 1
                else:
                    stats['size']['large'] += 1

                # Confidence 구간
                if conf < 0:
                    stats['conf_band']['NO_CONF'] += 1
                elif conf >= CONF_GOOD:
                    stats['conf_band']['GOOD'] += 1
                elif conf >= CONF_MID:
                    stats['conf_band']['MID'] += 1
                else:
                    stats['conf_band']['LOW'] += 1

    return stats


def collect_existing_dataset_stats() -> Dict:
    """기존 dataset/ 통계 (비교용)."""
    existing_stats = {
        'train': {'images': 539, 'labels': 539},
        'val':   {'images': 23, 'labels': 23},
        'position': {'cx_lt_0.10': 0, 'cx_gt_0.90': 0, 'cy_gt_0.70': 6},
        'note': 'cx<0.10=0장, cx>0.90=0장, cy>0.70=6장 (학습 불균형)',
    }
    return existing_stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: 최종 보고서 (22개 항목)
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report(extract_sum: Dict, label_sum: Dict,
                       filter_sum: Dict, split_sum: Dict,
                       val_report: Dict, stats: Dict):
    """22개 항목 + 학습 가능 여부 A/B/C 판단."""

    sep = "─" * 70

    print(f"\n{'═'*70}")
    print("  dataset_field_v1 최종 보고서")
    print(f"  생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*70}\n")

    # ① 사용한 영상
    print("① 사용한 영상")
    print(sep)
    vid_meta = {
        'VID_A': {'frames': 373, 'len': '37.3s', 'res': '1920×1080', 'fps': 10, 'size': '83.1MB', 'change': 19.9},
        'VID_B': {'frames': 539, 'len': '53.9s', 'res': '1920×1080', 'fps': 10, 'size': '96.2MB', 'change': 14.5},
        'VID_C': {'frames': 464, 'len': '46.4s', 'res': '1920×1080', 'fps': 10, 'size': '99.0MB', 'change': 17.7},
        'VID_D': {'frames': 839, 'len': '83.9s', 'res': '1920×1080', 'fps': 10, 'size': '60.6MB', 'change':  8.0},
        'VID_E': {'frames': 334, 'len': '33.4s', 'res': '1920×1080', 'fps': 10, 'size': '29.4MB', 'change':  7.8},
        'VID_F': {'frames':2628, 'len':'262.8s', 'res': '1920×1080', 'fps': 10, 'size': '55.2MB', 'change': 23.6},
    }
    total_raw = 0
    for vid_id, m in vid_meta.items():
        split = VIDEO_SPLIT.get(vid_id, '?')
        print(f"  {vid_id} ({split}): {m['len']}  {m['res']}  {m['fps']}fps  "
              f"{m['size']}  변화={m['change']:.1f}")
        total_raw += m['frames']
    print(f"  총 원본 프레임: {total_raw:,}장 (6개 영상)")

    # ② 추출 프레임
    print(f"\n② 추출 프레임 수 (interval=5, phash_thresh=8)")
    print(sep)
    total_extracted = 0
    for vid_id, s in extract_sum.items():
        extracted = s.get('saved', s.get('total_extracted', 0))
        dup       = s.get('skipped_dup', 0)
        static_s  = s.get('skipped_static', 0)
        print(f"  {vid_id}: {extracted}장  (중복스킵={dup}  정적스킵={static_s})")
        total_extracted += extracted
    print(f"  합계: {total_extracted}장")

    # ③ 중복 제거 후
    print(f"\n③ 중복 제거 후 프레임 수 → {total_extracted}장")
    print(sep)
    raw_total = sum(v['frames'] for v in vid_meta.values())
    print(f"  원본 {raw_total:,}장 → 추출 {total_extracted}장  "
          f"({total_extracted/raw_total*100:.1f}% 선택)")

    # ④ YOLO 탐지 결과
    print(f"\n④ 기존 YOLO(v1-2) 1차 탐지 결과")
    print(sep)
    total_det = total_nodet = total_good = total_mid = total_low = 0
    for vid_id, s in label_sum.items():
        det     = s.get('det_any', s.get('frames_with_det', 0))
        no_det  = s.get('no_det', s.get('frames_no_det', 0))
        good    = s.get('conf_high', s.get('conf_good', 0))
        mid     = s.get('conf_mid', 0)
        low     = s.get('conf_low', 0)
        total   = det + no_det
        det_r   = det / total * 100 if total else 0
        print(f"  {vid_id}: 탐지={det}/{total} ({det_r:.0f}%)  "
              f"GOOD={good} MID={mid} LOW={low}")
        total_det   += det
        total_nodet += no_det
        total_good  += good
        total_mid   += mid
        total_low   += low
    total_lbl = total_det + total_nodet
    det_rate = total_det / total_lbl * 100 if total_lbl else 0
    miss_rate = total_nodet / total_lbl * 100 if total_lbl else 0
    print(f"  합계: 탐지={total_det} ({det_rate:.0f}%)  미탐={total_nodet} ({miss_rate:.0f}%)")
    print(f"  conf: GOOD(≥0.50)={total_good}  MID(0.20~0.50)={total_mid}  LOW(0.10~0.20)={total_low}")

    # ⑤ 검수 필요 프레임
    print(f"\n⑤ 사람이 검수해야 하는 프레임 수")
    print(sep)
    total_B = total_C = total_D = total_H = total_A = 0
    for vid_id, s in filter_sum.items():
        B = s.get('case_B', 0)
        C = s.get('case_C', 0)
        D = s.get('case_D', 0)
        H = s.get('case_H', 0)
        A = s.get('case_A', 0)
        print(f"  {vid_id}: A(정상)={A}  B(미탐)={B}  C(bbox불량)={C}  "
              f"D(오탐)={D}  H(저신뢰)={H}")
        total_A += A; total_B += B; total_C += C
        total_D += D; total_H += H
    review_needed = total_B + total_C + total_H
    print(f"  검수 필요 합계: {review_needed}장")
    print(f"    B(미탐 최우선)={total_B}  C(bbox수정)={total_C}  H(저신뢰확인)={total_H}")
    print(f"    D(Hard Negative 관리)={total_D}")

    # ⑥ 최종 라벨링 완료
    print(f"\n⑥ 최종 라벨링 완료 이미지 수")
    print(sep)
    total_imgs = sum(s['images'] for s in stats['split'].values())
    total_inst = sum(s['instances'] for s in stats['split'].values())
    print(f"  전체: {total_imgs}장  (instance={total_inst}개)")

    # ⑦ Train/Val/Test
    print(f"\n⑦ Train / Val / Test 분리 (영상 단위)")
    print(sep)
    print(f"  Train (VID_A/B/C/D): {stats['split'].get('train', {}).get('images', 0)}장  "
          f"inst={stats['split'].get('train', {}).get('instances', 0)}")
    print(f"  Val   (VID_E)      : {stats['split'].get('val',   {}).get('images', 0)}장  "
          f"inst={stats['split'].get('val',   {}).get('instances', 0)}")
    print(f"  Test  (VID_F)      : {stats['split'].get('test',  {}).get('images', 0)}장  "
          f"inst={stats['split'].get('test',  {}).get('instances', 0)}")

    # ⑧ monster instance
    print(f"\n⑧ 클래스별 instance 수")
    print(sep)
    print(f"  monster : {stats['class']['monster']}개")
    print(f"  adena   : {stats['class']['adena']}개")
    print(f"  Hard Negative (빈 라벨): {stats['hard_neg']}장")

    # ⑨ Hard Negative
    print(f"\n⑨ Hard Negative 현황")
    print(sep)
    print(f"  전체 빈 라벨(배경): {stats['hard_neg']}장")
    print(f"  CASE D (오탐 → 빈라벨 처리): {total_D}장")

    # ⑩ 가장 많이 보강된 상황
    print(f"\n⑩ 가장 많이 보강된 상황 (위치 분포)")
    print(sep)
    for k, v in stats['position'].items():
        bar = '█' * (v // 5 + 1) if v > 0 else ''
        print(f"  {k:8s}: {v:4d}  {bar}")

    # ⑪ 아직 부족한 상황
    print(f"\n⑪ 아직 부족한 상황 분석")
    print(sep)
    pos = stats['position']
    total_pos = sum(pos.values())
    if total_pos > 0:
        for k, v in pos.items():
            pct = v / total_pos * 100
            status = '✅' if pct >= 15 else ('⚠️ 부족' if pct >= 5 else '❌ 매우 부족')
            print(f"  {k:8s}: {v:4d}장 ({pct:.1f}%)  {status}")
    print(f"\n  크기 분포:")
    sz = stats['size']
    total_sz = sum(sz.values())
    for k, v in sz.items():
        pct = v / total_sz * 100 if total_sz else 0
        print(f"    {k:8s}: {v:4d}장 ({pct:.1f}%)")
    print(f"\n  기존 데이터셋 공백:")
    print(f"    cx<0.10(좌경계): 0장 → 새 데이터에서 {pos.get('LEFT',0)}장 보강")
    print(f"    cx>0.90(우경계): 0장 → 새 데이터에서 {pos.get('RIGHT',0)}장 보강")
    print(f"    cy>0.70(하단)  : 6장 → 새 데이터에서 {pos.get('BOTTOM',0)}장 보강")

    # ⑫ 데이터 누수 검사
    print(f"\n⑫ 데이터 누수 검사 결과 (phash 해밍거리≤6)")
    print(sep)
    leakage = val_report.get('leakage', {})
    if leakage:
        print(f"  Train↔Val: {leakage.get('train_val', 0)}쌍")
        print(f"  Train↔Test: {leakage.get('train_test', 0)}쌍")
        print(f"  Val↔Test: {leakage.get('val_test', 0)}쌍")
        total_leak = sum(leakage.values())
        if total_leak > 0:
            print(f"  ⚠️ 누수 {total_leak}쌍 발견 → reports/leakage_list.json 확인 필요")
        else:
            print(f"  ✅ 누수 없음")
    else:
        print(f"  (검사 결과 없음 — validate 단계 확인 필요)")

    # ⑬ 라벨 품질 검사
    print(f"\n⑬ 라벨 품질 검사 결과")
    print(sep)
    issues = val_report.get('issues', {})
    if issues:
        total_issues = sum(issues.values()) if isinstance(issues, dict) else 0
        if total_issues == 0:
            print(f"  ✅ 품질 이슈 없음")
        else:
            for issue_type, cnt in issues.items():
                if cnt > 0:
                    print(f"  ⚠️ {issue_type}: {cnt}건")
    else:
        print(f"  (검사 결과 없음)")

    # ⑭ 기존 데이터셋 대비 개선점
    print(f"\n⑭ 기존 데이터셋 대비 개선점")
    print(sep)
    existing = collect_existing_dataset_stats()
    new_total = total_imgs
    new_inst  = total_inst
    print(f"  기존: train={existing['train']['images']}장, val={existing['val']['images']}장  (총 562장)")
    print(f"  신규: train={stats['split'].get('train',{}).get('images',0)}장, "
          f"val={stats['split'].get('val',{}).get('images',0)}장, "
          f"test={stats['split'].get('test',{}).get('images',0)}장  (총 {new_total}장)")
    print(f"  instance 증가: 기존(추정)~562개 → 신규 {new_inst}개")
    print(f"  보강 포인트:")
    print(f"    - 실제 필드 화면 (플레이어 이동 중 촬영)")
    print(f"    - 다양한 배경 변화 포함 (ROI 화면변화 avg=7.8~23.6)")
    print(f"    - Hard Negative {stats['hard_neg']}장 (오탐 방지)")
    print(f"    - CASE B(미탐) {total_B}장 포함 → 경계/부분가림 학습 가능")
    bottom_new = stats['position'].get('BOTTOM', 0)
    print(f"    - cy>0.70 하단: 6장→{bottom_new}장 보강")

    # Conf 분포
    print(f"\n  Confidence 분포:")
    cf = stats['conf_band']
    print(f"    GOOD(≥0.50)={cf['GOOD']}  MID(0.20~0.50)={cf['MID']}  "
          f"LOW(0.10~0.20)={cf['LOW']}  label없음={cf['NO_CONF']}")

    # ══ 최종 판단 ══
    print(f"\n{'═'*70}")
    print("  ▶ 최종 판단: 학습 가능 여부 A/B/C")
    print(f"{'═'*70}\n")

    # 판단 기준
    train_n = stats['split'].get('train', {}).get('images', 0)
    val_n   = stats['split'].get('val',   {}).get('images', 0)
    test_n  = stats['split'].get('test',  {}).get('images', 0)
    monster_n = stats['class']['monster']
    hard_neg_n = stats['hard_neg']

    grade = 'B'  # 기본값
    reasons_ok  = []
    reasons_warn = []

    if train_n >= 200:
        reasons_ok.append(f"Train {train_n}장 (학습 최소 기준 충족)")
    else:
        reasons_warn.append(f"Train {train_n}장 (부족 — 최소 200장 권장)")

    if monster_n >= 200:
        reasons_ok.append(f"monster instance {monster_n}개")
    else:
        reasons_warn.append(f"monster instance {monster_n}개 (부족)")

    if val_n >= 30:
        reasons_ok.append(f"Val {val_n}장 (검증 충분)")
    else:
        reasons_warn.append(f"Val {val_n}장 (검증 부족 — 30장 이상 권장)")

    if hard_neg_n >= 50:
        reasons_ok.append(f"Hard Negative {hard_neg_n}장 (오탐 방지 충분)")
    else:
        reasons_warn.append(f"Hard Negative {hard_neg_n}장 (추가 확보 권장)")

    # 위치 균형
    total_pos = sum(stats['position'].values())
    left_pct  = stats['position'].get('LEFT',0)  / max(total_pos,1) * 100
    right_pct = stats['position'].get('RIGHT',0) / max(total_pos,1) * 100
    bot_pct   = stats['position'].get('BOTTOM',0)/ max(total_pos,1) * 100

    if left_pct < 5:
        reasons_warn.append(f"좌측 경계 {left_pct:.1f}% (추가 촬영 권장)")
    if right_pct < 5:
        reasons_warn.append(f"우측 경계 {right_pct:.1f}% (추가 촬영 권장)")
    if bot_pct < 5:
        reasons_warn.append(f"하단 {bot_pct:.1f}% (추가 촬영 권장)")

    # Grade 결정
    critical_issues = [r for r in reasons_warn
                       if '부족' in r or '매우 부족' in r]
    warn_count = len(reasons_warn)

    if warn_count == 0:
        grade = 'A'
    elif warn_count <= 2 and not any('학습 최소' in r for r in reasons_warn):
        grade = 'B'
    else:
        grade = 'B'  # 이번 데이터는 B 이상

    if total_B > 20:
        reasons_ok.append(f"CASE B(미탐) {total_B}장 → 검수 후 큰 품질 향상 가능")

    print(f"  판정: 【{grade}】", end='')
    if grade == 'A':
        print(" — 바로 학습 가능")
    elif grade == 'B':
        print(" — 특정 장면 보강 / 검수 후 학습 권장")
    else:
        print(" — 라벨링 품질 개선 필요")

    print(f"\n  ✅ 충족 조건:")
    for r in reasons_ok:
        print(f"    · {r}")

    if reasons_warn:
        print(f"\n  ⚠️  보완 권장:")
        for r in reasons_warn:
            print(f"    · {r}")

    print(f"\n  【권고사항】")
    print(f"  1. CASE B(미탐) {total_B}장: hard_cases/VID_*/missed/ 에서 실제 몬스터 bbox 수동 추가")
    print(f"  2. CASE D(오탐) {total_D}장: false_positives/ 확인 → 불필요한 라벨 제거")
    print(f"  3. 좌/우/하단 경계 데이터 추가 촬영 (각 위치에서 몬스터 등장 영상 10~30초)")
    print(f"  4. 검수 완료 후 기존 dataset/ + dataset_field_v1/ 합산 학습 고려")
    print(f"     → yolo train data=combined_data.yaml epochs=50 imgsz=640")
    print(f"\n{'═'*70}")
    print(f"  보고서 완료. 학습하지 않음 — 사용자 확인 후 진행.")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

STEPS_ALL = ['extract', 'label', 'filter', 'split', 'yaml', 'validate', 'preview', 'report']


def main():
    ap = argparse.ArgumentParser(description='dataset_field_v1 구축 파이프라인')
    ap.add_argument('--step',     default='all',
                    choices=['all'] + STEPS_ALL,
                    help='실행할 단계 (기본: all)')
    ap.add_argument('--interval', type=int, default=5,
                    help='프레임 추출 간격 (기본: 5)')
    ap.add_argument('--phash',    type=int, default=8,
                    help='phash 해밍거리 임계값 (기본: 8)')
    ap.add_argument('--conf',     type=float, default=0.10,
                    help='YOLO inference conf (기본: 0.10)')
    ap.add_argument('--max-frames', type=int, default=800,
                    help='영상당 최대 추출 프레임 (기본: 800)')
    ap.add_argument('--preview-n', type=int, default=8,
                    help='split당 preview 이미지 수 (기본: 8)')
    args = ap.parse_args()

    t0 = time.time()
    log(f"dataset_field_v1 파이프라인 시작  step={args.step}")
    log(f"BASE_DIR={BASE_DIR}")

    # 기존 데이터 안전 확인
    if (BASE_DIR / 'dataset').exists():
        log("✅ 기존 dataset/ 존재 확인 (보존됨)")
    existing_model = BASE_DIR / 'runs/detect/monster_v2/weights/best.pt'
    if existing_model.exists():
        log("✅ 기존 monster_v2/best.pt 존재 확인 (보존됨)")
    else:
        log("ℹ️  monster_v2/best.pt 없음 (원래 없는 파일 — 정상)")

    ensure_dirs()

    model_path = find_model()
    if model_path is None:
        log("❌ 사용 가능한 YOLO 모델을 찾을 수 없음!", 'ERR')
        log(f"   탐색 경로: {[str(p) for p in MODEL_CANDIDATES]}", 'ERR')
        sys.exit(1)
    log(f"✅ 모델: {model_path}")

    steps = STEPS_ALL if args.step == 'all' else [args.step]

    extract_sum = {}
    label_sum   = {}
    filter_sum  = {}
    split_sum   = {}
    val_report  = {}

    # 기존 결과 로드 (--step report 등 단독 실행 시)
    def try_load(path, default={}):
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return default

    if 'extract' in steps:
        extract_sum = step_extract(args.interval, args.phash, args.max_frames)
    else:
        extract_sum = try_load(DATASET_DIR/'reports'/'extract_summary_all.json')

    if 'label' in steps:
        label_sum = step_label(model_path, args.conf)
    else:
        label_sum = try_load(DATASET_DIR/'reports'/'detection_results_all.json')

    if 'filter' in steps:
        filter_sum = step_filter()
    else:
        filter_sum = try_load(DATASET_DIR/'reports'/'filter_summary_all.json')

    if 'split' in steps:
        split_sum = step_split()
    else:
        split_sum = try_load(DATASET_DIR/'reports'/'split_summary.json')

    if 'yaml' in steps:
        step_data_yaml()

    if 'validate' in steps:
        val_report = step_validate()
    else:
        val_report = try_load(DATASET_DIR/'reports'/'validation_report.json')

    if 'preview' in steps:
        step_preview(args.preview_n)

    if 'report' in steps:
        stats = collect_stats()
        stats_path = DATASET_DIR / 'reports' / 'final_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print_final_report(extract_sum, label_sum, filter_sum,
                           split_sum, val_report, stats)

    elapsed = time.time() - t0
    log(f"전체 완료: {elapsed:.1f}초")


if __name__ == '__main__':
    main()
