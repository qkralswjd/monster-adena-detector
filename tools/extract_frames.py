"""
tools/extract_frames.py
=======================
영상에서 프레임을 추출하고 perceptual hash 기반으로 중복을 제거한다.

특징:
  - ROI(게임 필드 영역)만 기준으로 유사도 판단
  - phash 해밍거리 기반 중복 제거
  - 화면 변화량 기반 적응형 샘플링 (정적 장면 스킵)
  - 전체 프레임 추출 후 영상 단위 분리 정보 유지

사용법:
  python tools/extract_frames.py \
      --videos test_runs/*/screen*.mp4 \
      --out    dataset_field_v1/frames \
      --interval 5 \
      --phash-thresh 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── ROI 설정 (config.json 기준) ──────────────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472

# ── phash 파라미터 ───────────────────────────────────────────────────────────
PHASH_SIZE   = 16    # DCT 크기
PHASH_HIGH_F = 4     # 저주파 성분 크기

# ── 화면 변화 임계값 ─────────────────────────────────────────────────────────
SCENE_CHANGE_THRESH = 40.0   # 이 이상이면 장면 전환으로 무조건 포함
STATIC_SKIP_THRESH  = 0.3    # 이 이하면 완전 정적 (로딩/메뉴 화면만 스킵)


# ══════════════════════════════════════════════════════════════════════════════
# perceptual hash
# ══════════════════════════════════════════════════════════════════════════════

def compute_phash(roi_bgr: np.ndarray) -> np.ndarray:
    """ROI BGR 이미지 → 64비트 phash 벡터."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:PHASH_HIGH_F, :PHASH_HIGH_F].flatten()
    med = np.median(dct_low)
    return (dct_low > med).astype(np.uint8)


def hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


# ══════════════════════════════════════════════════════════════════════════════
# 프레임 추출 (단일 영상)
# ══════════════════════════════════════════════════════════════════════════════

def extract_from_video(
    video_path: str,
    out_dir: str,
    video_id: str,
    interval: int = 5,
    phash_thresh: int = 8,
    max_frames: int = 1000,
) -> Dict:
    """
    단일 영상에서 프레임 추출.

    Returns:
        {
          'video_id': str,
          'total_frames': int,
          'sampled': int,
          'after_dedup': int,
          'frames': [{'path': str, 'frame_no': int, 'ts_sec': float, ...}]
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 열 수 없음: {video_path}")
        return {}

    fps       = cap.get(cv2.CAP_PROP_FPS)
    fc        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = fc / fps if fps > 0 else 0

    print(f"\n[EXTRACT] {video_id}: {vid_w}x{vid_h}  {fps:.1f}fps  "
          f"{fc}frames  {duration:.1f}s")

    os.makedirs(out_dir, exist_ok=True)

    # ── 1단계: interval마다 샘플링 ──────────────────────────────────────────
    candidates = []
    prev_roi_gray = None
    fn = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 화면 변화량
        if prev_roi_gray is not None:
            diff = float(np.mean(np.abs(roi_gray.astype(np.float32)
                                        - prev_roi_gray.astype(np.float32))))
        else:
            diff = 99.0

        phash = compute_phash(roi)

        candidates.append({
            'frame_no': fn,
            'ts_sec':   round(fn / fps, 2) if fps > 0 else 0,
            'diff':     round(diff, 2),
            'phash':    phash,
            'roi':      roi.copy(),
            'frame':    frame.copy(),
        })

        prev_roi_gray = roi_gray
        fn += interval
        if fn >= fc:
            break

    cap.release()
    sampled = len(candidates)
    print(f"  → 샘플링: {sampled}프레임 (interval={interval})")

    # ── 2단계: phash 중복 제거 ──────────────────────────────────────────────
    kept = []
    hashes_kept = []

    for cand in candidates:
        ph = cand['phash']
        diff = cand['diff']

        # 완전 정적 프레임 스킵 (메뉴/로딩 화면 등)
        if diff < STATIC_SKIP_THRESH and kept:
            continue

        # 장면 전환 → 무조건 포함
        if diff >= SCENE_CHANGE_THRESH:
            kept.append(cand)
            hashes_kept.append(ph)
            continue

        # phash 유사도 검사
        is_dup = False
        for ph_prev in hashes_kept[-20:]:  # 최근 20개와만 비교 (속도)
            if hamming(ph, ph_prev) <= phash_thresh:
                is_dup = True
                break

        if not is_dup:
            kept.append(cand)
            hashes_kept.append(ph)

    print(f"  → 중복 제거: {sampled} → {len(kept)}")

    # ── 3단계: 이미지 저장 ──────────────────────────────────────────────────
    frame_records = []
    for i, cand in enumerate(kept):
        fname = f"{video_id}_f{cand['frame_no']:06d}.jpg"
        save_path = os.path.join(out_dir, fname)

        # 전체 1920×1080 저장 (나중에 ROI crop은 라벨링 시)
        cv2.imwrite(save_path, cand['frame'], [cv2.IMWRITE_JPEG_QUALITY, 92])

        frame_records.append({
            'fname':    fname,
            'path':     save_path,
            'video_id': video_id,
            'frame_no': cand['frame_no'],
            'ts_sec':   cand['ts_sec'],
            'diff':     cand['diff'],
        })

    return {
        'video_id':     video_id,
        'video_path':   video_path,
        'fps':          fps,
        'total_frames': fc,
        'duration_sec': round(duration, 1),
        'width':        vid_w,
        'height':       vid_h,
        'sampled':      sampled,
        'after_dedup':  len(kept),
        'frames':       frame_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 전체 영상 처리
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    # 영상 목록 수집
    video_paths = []
    for pattern in args.videos:
        video_paths.extend(sorted(glob.glob(pattern)))
    video_paths = sorted(set(video_paths))

    if not video_paths:
        print("[ERROR] 영상 파일 없음")
        sys.exit(1)

    print(f"[INFO] 처리할 영상: {len(video_paths)}개")
    for vp in video_paths:
        print(f"  {vp}")

    os.makedirs(args.out, exist_ok=True)

    all_results = []
    for i, vp in enumerate(video_paths):
        # VID_A, VID_B, ... 형식 ID
        vid_id = f"VID_{chr(65 + i)}"
        result = extract_from_video(
            video_path   = vp,
            out_dir      = args.out,
            video_id     = vid_id,
            interval     = args.interval,
            phash_thresh = args.phash_thresh,
            max_frames   = getattr(args, 'max_frames', 1000),
        )
        if result:
            all_results.append(result)

    # ── 요약 저장 ────────────────────────────────────────────────────────────
    summary = {
        'videos': [],
        'total_extracted': 0,
    }
    for r in all_results:
        summary['videos'].append({
            'video_id':     r['video_id'],
            'video_path':   r['video_path'],
            'fps':          r['fps'],
            'total_frames': r['total_frames'],
            'duration_sec': r['duration_sec'],
            'width':        r['width'],
            'height':       r['height'],
            'sampled':      r['sampled'],
            'after_dedup':  r['after_dedup'],
        })
        summary['total_extracted'] += r['after_dedup']

    meta_path = os.path.join(args.out, 'extract_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  프레임 추출 완료")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['video_id']}: {r['sampled']}→{r['after_dedup']}장  "
              f"({r['duration_sec']:.0f}초 영상)")
    print(f"  총 추출: {summary['total_extracted']}장")
    print(f"  저장 위치: {args.out}/")
    print(f"  메타: {meta_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='영상 프레임 추출 + phash 중복 제거')
    parser.add_argument('--videos', nargs='+', required=True,
                        help='영상 파일 경로 (glob 가능)')
    parser.add_argument('--out', default='dataset_field_v1/frames',
                        help='추출 프레임 저장 디렉토리')
    parser.add_argument('--interval', type=int, default=5,
                        help='프레임 추출 간격 (기본: 5프레임 = 0.5초)')
    parser.add_argument('--phash-thresh', type=int, default=2,
                        help='phash 해밍거리 임계값 (클수록 더 많이 제거, 기본: 2)'
                             ' — 16비트 phash 기준, 2=거의동일만제거, 8=비슷하면모두제거')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='영상당 최대 추출 프레임 수 (기본: 1000)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
