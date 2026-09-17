"""
tools/build_dataset_A.py
========================
CASE A (고신뢰 탐지 성공) 프레임만으로 dataset_field_A_v1 구축.

원칙:
  - dataset_field_v1 수정/삭제 없음 (B/C 보존)
  - CASE A만 사용 (가장 확실한 GT)
  - min_gap=9 stride 적용으로 연속 중복 프레임 감소
  - 영상 단위 train/val/test 분리 유지
  - 이미지 심볼릭 링크 대신 실제 복사
  - YOLO format 라벨 .txt 생성
  - 검수용 contact sheet (전체 + 영상별) 생성

출력:
  dataset_field_A_v1/
  ├── images/{train,val,test}/
  ├── labels/{train,val,test}/
  ├── previews/{train,val,test}/
  ├── contact_sheets/
  ├── reports/
  └── data.yaml
"""

from __future__ import annotations
import cv2, json, math, random, shutil, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ─── 경로 ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
SRC        = BASE / 'dataset_field_v1'          # 원본 (읽기 전용)
DST        = BASE / 'dataset_field_A_v1'        # 새 데이터셋
MODEL_PATH = BASE / 'runs/detect/monster_v1-2/weights/best.pt'

# ─── ROI ─────────────────────────────────────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472

# ─── 설정 ────────────────────────────────────────────────────────────────────
MIN_GAP   = 9      # 연속 프레임 최소 간격 (frame 단위, 10fps → 0.9초)
RAND_SEED = 42

VID_ORDER = ['VID_A', 'VID_B', 'VID_C', 'VID_D', 'VID_E', 'VID_F']
VIDS_INFO = {
    'VID_A': 'train', 'VID_B': 'train', 'VID_C': 'train',
    'VID_D': 'val',   'VID_E': 'val',   'VID_F': 'test',
}

# ─── 색상 (BGR) ──────────────────────────────────────────────────────────────
C_GT    = (0, 220, 0)      # 초록 - GT bbox
C_YOLO  = (0, 200, 255)    # 하늘색 - YOLO bbox
C_INFO  = (200, 200, 200)
INFO_H  = 44

# ─── 썸네일 크기 ─────────────────────────────────────────────────────────────
THUMB_W  = 540
THUMB_H  = 224
COLS     = 4
PAGE_SZ  = 20   # contact sheet 당 장 수


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ─── CASE A 프레임 로드 ───────────────────────────────────────────────────────
def load_case_A() -> Dict[str, List[dict]]:
    """영상별 CASE A 프레임 목록 반환. {vid_id: [frame_record, ...]}"""
    by_vid: Dict[str, List[dict]] = defaultdict(list)
    for vid_id in VID_ORDER:
        cache = SRC / 'reports' / f'{vid_id}_gt_results.json'
        if not cache.exists():
            log(f"  [WARN] 캐시 없음: {cache}")
            continue
        with open(cache) as f:
            raw = json.load(f)
        for k, fr in raw.items():
            if fr['case'] == 'A':
                fr['vid_id']   = vid_id
                fr['split']    = VIDS_INFO[vid_id]
                fr['frame_no'] = int(k)
                by_vid[vid_id].append(fr)
        by_vid[vid_id].sort(key=lambda x: x['frame_no'])
    return by_vid


# ─── stride 필터링 ────────────────────────────────────────────────────────────
def apply_stride(frames: List[dict], min_gap: int) -> List[dict]:
    """
    연속 프레임 중복 제거.
    frame_no 기준으로 이전 선택 프레임과의 간격이 min_gap 미만이면 스킵.
    단, 구간 변화(gap > min_gap*5)가 생기면 무조건 다음 프레임 포함.
    """
    if not frames:
        return []
    result = [frames[0]]
    for fr in frames[1:]:
        gap = fr['frame_no'] - result[-1]['frame_no']
        if gap >= min_gap:
            result.append(fr)
    return result


# ─── YOLO 라벨 저장 ───────────────────────────────────────────────────────────
def save_yolo_label(fr: dict, label_path: Path) -> int:
    """
    gt_dets → YOLO format .txt.
    CASE A 라벨만 사용 (cls_name=monster → 0, adena → 1).
    반환: bbox 수
    """
    lines = []
    for det in fr.get('gt_dets', []):
        cls_id = det.get('cls_id', 0)
        cx = det['cx_n']
        cy = det['cy_n']
        w  = det['w_n']
        h  = det['h_n']
        # 범위 클램프
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if lines:
        label_path.write_text('\n'.join(lines) + '\n')
    else:
        label_path.write_text('')   # 빈 라벨 (EMPTY)
    return len(lines)


# ─── Preview 이미지 생성 ─────────────────────────────────────────────────────
def draw_preview(fr: dict, out_path: Path) -> bool:
    src_path = Path(fr['path'])
    if not src_path.exists():
        return False
    img = cv2.imread(str(src_path))
    if img is None:
        return False

    H, W = img.shape[:2]
    roi = img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W].copy()
    rh, rw = roi.shape[:2]

    # YOLO bbox (하늘색 점선)
    for det in fr.get('yolo_dets', []):
        x1, y1 = int(det['x1r']), int(det['y1r'])
        x2, y2 = int(det['x2r']), int(det['y2r'])
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(rw-1, x2); y2c = min(rh-1, y2)
        if x2c > x1c and y2c > y1c:
            _dashed_rect(roi, x1c, y1c, x2c, y2c, C_YOLO, 2)
            cv2.putText(roi, f"Y:{det.get('conf',0):.2f}",
                        (x1c+2, y1c-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, C_YOLO, 1, cv2.LINE_AA)

    # GT bbox (초록 실선)
    for det in fr.get('gt_dets', []):
        x1, y1 = int(det['x1r']), int(det['y1r'])
        x2, y2 = int(det['x2r']), int(det['y2r'])
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(rw-1, x2); y2c = min(rh-1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        cv2.rectangle(roi, (x1c, y1c), (x2c, y2c), C_GT, 3)
        conf = det.get('conf', 0)
        tlen = det.get('track_len', 0)
        lbl  = f"A {conf:.2f} tr:{tlen}"
        lw   = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.rectangle(roi, (x1c, max(0, y1c-20)), (x1c+lw+4, y1c), (0,0,0), -1)
        cv2.putText(roi, lbl, (x1c+2, y1c-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GT, 1, cv2.LINE_AA)

    # OK 배지
    cv2.rectangle(roi, (rw-54, 4), (rw-4, 26), (0, 160, 0), -1)
    cv2.putText(roi, 'CASE A', (rw-52, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # 정보 바
    info = np.zeros((INFO_H, rw, 3), dtype=np.uint8)
    info[:] = (25, 25, 25)
    vid_id = fr.get('vid_id','?')
    split  = fr.get('split','?')
    fn     = fr.get('frame_no', 0)
    ts     = fr.get('ts_sec', 0.0)
    n_gt   = len(fr.get('gt_dets', []))
    cv2.putText(info, f"{vid_id}[{split}] f{fn:06d} t={ts:.1f}s",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_INFO, 1, cv2.LINE_AA)
    cv2.putText(info, f"CASE=A  GT={n_gt}box  gap-filtered",
                (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GT, 1, cv2.LINE_AA)

    combined = np.vstack([roi, info])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def _dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash=8):
    def dl(p1, p2):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        L = math.sqrt(dx*dx + dy*dy)
        if L == 0: return
        n = max(1, int(L/dash))
        for i in range(n):
            if i % 2 == 1: continue
            t0, t1 = i/n, min(1.0, (i+0.8)/n)
            cv2.line(img,
                     (int(p1[0]+dx*t0), int(p1[1]+dy*t0)),
                     (int(p1[0]+dx*t1), int(p1[1]+dy*t1)),
                     color, thickness)
    dl((x1,y1),(x2,y1)); dl((x2,y1),(x2,y2))
    dl((x2,y2),(x1,y2)); dl((x1,y2),(x1,y1))


# ─── Contact Sheet ────────────────────────────────────────────────────────────
def make_contact_sheet(prev_paths: List[Path], out_path: Path,
                       title: str, cols: int = COLS,
                       thumb_w: int = THUMB_W, thumb_h: int = THUMB_H) -> bool:
    if not prev_paths:
        return False
    cell_h = thumb_h + INFO_H
    rows   = math.ceil(len(prev_paths) / cols)
    sheet  = np.zeros((50 + rows * cell_h, cols * thumb_w, 3), dtype=np.uint8)
    sheet[:] = (20, 20, 20)
    cv2.putText(sheet, title, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220,220,220), 2, cv2.LINE_AA)

    for idx, p in enumerate(prev_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, cell_h))
        r, c  = divmod(idx, cols)
        y0, x0 = 50 + r * cell_h, c * thumb_w
        sheet[y0:y0+cell_h, x0:x0+thumb_w] = thumb
        cv2.rectangle(sheet, (x0, y0), (x0+thumb_w-1, y0+cell_h-1), (70,70,70), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


# ─── 통계 계산 ────────────────────────────────────────────────────────────────
def calc_stats(frames: List[dict]) -> dict:
    pos  = defaultdict(int)
    size = defaultdict(int)
    bbox_total = 0
    for fr in frames:
        for det in fr.get('gt_dets', []):
            if det.get('cls_name') != 'monster':
                continue
            bbox_total += 1
            cx, cy = det['cx_n'], det['cy_n']
            w,  h  = det['w_n'],  det['h_n']
            area   = w * h
            if cx < 0.33:   pos['LEFT']   += 1
            elif cx < 0.67: pos['CENTER'] += 1
            else:            pos['RIGHT']  += 1
            if cy < 0.33:   pos['TOP']    += 1
            elif cy < 0.67: pos['MIDDLE'] += 1
            else:            pos['BOTTOM'] += 1
            if area < 0.005:   size['SMALL']  += 1
            elif area < 0.02:  size['MEDIUM'] += 1
            else:              size['LARGE']   += 1

    # 연속 프레임 통계
    by_vid: Dict[str, List[int]] = defaultdict(list)
    for fr in frames:
        by_vid[fr['vid_id']].append(fr['frame_no'])
    max_streak = 0
    for fnums in by_vid.values():
        fnums = sorted(fnums)
        cur = 1
        for i in range(1, len(fnums)):
            if fnums[i] - fnums[i-1] <= 6:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 1

    return {
        'total_frames': len(frames),
        'bbox_total': bbox_total,
        'position': dict(pos),
        'size': dict(size),
        'max_consecutive': max_streak,
    }


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 65)
    print("  🏗  dataset_field_A_v1 구축 (CASE A only, stride=9)")
    print("=" * 65)

    # 디렉토리 생성
    for split in ('train', 'val', 'test'):
        (DST / 'images'   / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels'   / split).mkdir(parents=True, exist_ok=True)
        (DST / 'previews' / split).mkdir(parents=True, exist_ok=True)
    (DST / 'contact_sheets').mkdir(parents=True, exist_ok=True)
    (DST / 'reports').mkdir(parents=True, exist_ok=True)

    # ── STEP 1: CASE A 로드 ───────────────────────────────────────────────────
    log("STEP 1: CASE A 로드")
    by_vid_raw = load_case_A()
    raw_total  = sum(len(v) for v in by_vid_raw.values())
    log(f"  원본 CASE A: {raw_total}장")
    for vid_id in VID_ORDER:
        n = len(by_vid_raw.get(vid_id, []))
        log(f"    {vid_id}[{VIDS_INFO[vid_id]}]: {n}장")

    # ── STEP 2: stride 필터링 ─────────────────────────────────────────────────
    log(f"\nSTEP 2: stride 필터링 (min_gap={MIN_GAP})")
    by_vid_filtered: Dict[str, List[dict]] = {}
    all_filtered: List[dict] = []

    for vid_id in VID_ORDER:
        raw   = by_vid_raw.get(vid_id, [])
        filt  = apply_stride(raw, MIN_GAP)
        by_vid_filtered[vid_id] = filt
        all_filtered.extend(filt)
        log(f"    {vid_id}: {len(raw)}장 → {len(filt)}장 "
            f"({len(raw)-len(filt)}장 제거)")

    filt_total = len(all_filtered)
    log(f"  stride 후: {filt_total}장 (제거: {raw_total-filt_total}장, "
        f"유지율: {filt_total/raw_total*100:.1f}%)")

    # split별 집계
    split_frames: Dict[str, List[dict]] = defaultdict(list)
    for fr in all_filtered:
        split_frames[fr['split']].append(fr)
    for sp in ('train', 'val', 'test'):
        log(f"    {sp}: {len(split_frames[sp])}장")

    # ── STEP 3: 이미지 복사 + 라벨 생성 ─────────────────────────────────────
    log("\nSTEP 3: 이미지 복사 + YOLO 라벨 생성")
    stats_by_split: Dict[str, dict] = {}
    all_prev_paths: List[Path]      = []
    prev_by_vid:    Dict[str, List[Path]] = defaultdict(list)

    for split in ('train', 'val', 'test'):
        frames   = split_frames[split]
        img_dir  = DST / 'images' / split
        lbl_dir  = DST / 'labels' / split
        prev_dir = DST / 'previews' / split

        n_img = 0; n_bbox = 0; n_empty = 0
        for fr in frames:
            src_img = Path(fr['path'])
            if not src_img.exists():
                continue

            fname = src_img.name
            # 이미지 복사
            dst_img = img_dir / fname
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # 라벨 저장
            lbl_path = lbl_dir / (src_img.stem + '.txt')
            nb = save_yolo_label(fr, lbl_path)
            n_bbox += nb
            if nb == 0:
                n_empty += 1

            # Preview 생성
            prev_path = prev_dir / (src_img.stem + '_prev.jpg')
            if draw_preview(fr, prev_path):
                all_prev_paths.append(prev_path)
                prev_by_vid[fr['vid_id']].append(prev_path)

            n_img += 1

        stats_by_split[split] = {
            'images': n_img, 'bbox': n_bbox, 'empty': n_empty
        }
        log(f"    [{split}] images={n_img}  bbox={n_bbox}  empty_label={n_empty}")

    # ── STEP 4: Contact Sheet 생성 ────────────────────────────────────────────
    log("\nSTEP 4: Contact Sheet 생성")
    cs_dir = DST / 'contact_sheets'

    # 전체 (랜덤 샘플 최대 PAGE_SZ장)
    random.seed(RAND_SEED)
    overall_sample = random.sample(all_prev_paths, min(PAGE_SZ, len(all_prev_paths)))
    overall_sample.sort()
    make_contact_sheet(overall_sample, cs_dir / '00_overall.jpg',
                       f"dataset_field_A_v1 — 전체 샘플 ({len(overall_sample)}장, stride={MIN_GAP})",
                       cols=COLS)
    log(f"    00_overall.jpg ({len(overall_sample)}장)")

    # train / val / test 각각
    for split in ('train', 'val', 'test'):
        paths = sorted(p for p in all_prev_paths
                       if split_frames[split] and
                       any(p.stem.startswith(fr['vid_id'])
                           for fr in split_frames[split]))
        # split에 맞는 preview만 필터
        split_paths = []
        split_fnames = {Path(fr['path']).stem for fr in split_frames[split]}
        for p in all_prev_paths:
            if p.stem.replace('_prev','') in split_fnames:
                split_paths.append(p)

        sample = split_paths[:PAGE_SZ]
        if sample:
            out_p = cs_dir / f'01_{split}.jpg'
            make_contact_sheet(
                sample, out_p,
                f"CASE A [{split}] — {len(split_paths)}장 중 {len(sample)}장",
                cols=COLS)
            log(f"    01_{split}.jpg ({len(sample)}장)")

    # 영상별 contact sheet (각 영상 대표 8장)
    for vid_id in VID_ORDER:
        paths = prev_by_vid.get(vid_id, [])
        if not paths:
            continue
        sample = paths[:8]
        out_p  = cs_dir / f'02_{vid_id}.jpg'
        make_contact_sheet(
            sample, out_p,
            f"CASE A [{vid_id}/{VIDS_INFO[vid_id]}] — {len(paths)}장 중 {len(sample)}장",
            cols=4)
        log(f"    02_{vid_id}.jpg ({len(sample)}장)")

    # ── STEP 5: data.yaml ─────────────────────────────────────────────────────
    log("\nSTEP 5: data.yaml 생성")
    yaml = f"""# dataset_field_A_v1
# CASE A only (고신뢰 탐지 + 트랙 확정)
# stride={MIN_GAP} 적용 (연속 중복 프레임 감소)
# 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 분할: Train=VID_A/B/C  Val=VID_D/E  Test=VID_F
# 원본 dataset_field_v1 보존 (B/C 라벨 삭제하지 않음)

path: {DST.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: monster
  1: adena
"""
    (DST / 'data.yaml').write_text(yaml)
    log(f"    {DST/'data.yaml'}")

    # ── STEP 6: 품질 검사 ─────────────────────────────────────────────────────
    log("\nSTEP 6: 라벨 품질 검사")
    issues_total = 0
    for split in ('train', 'val', 'test'):
        img_dir = DST / 'images' / split
        lbl_dir = DST / 'labels' / split
        imgs    = set(p.stem for p in img_dir.glob('*.jpg'))
        lbls    = set(p.stem for p in lbl_dir.glob('*.txt'))
        missing = imgs - lbls
        orphan  = lbls - imgs
        bad     = []
        for lbl_p in lbl_dir.glob('*.txt'):
            for line in lbl_p.read_text().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad.append(lbl_p.name); break
                try:
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    if not (0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1):
                        bad.append(lbl_p.name); break
                except:
                    bad.append(lbl_p.name); break
        issues = len(missing) + len(orphan) + len(bad)
        issues_total += issues
        log(f"    [{split}] images={len(imgs)}  labels={len(lbls)}  "
            f"missing={len(missing)}  orphan={len(orphan)}  "
            f"format_err={len(bad)}  issues={issues}")

    # ── 최종 통계 + 보고서 JSON ───────────────────────────────────────────────
    log("\nSTEP 7: 최종 보고서")
    raw_stats  = calc_stats(sum(by_vid_raw.values(), []))
    filt_stats = calc_stats(all_filtered)

    report = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {'case': 'A_only', 'min_gap': MIN_GAP},
        'raw':    {'total': raw_total,
                   'by_vid': {v: len(by_vid_raw.get(v,[])) for v in VID_ORDER}},
        'filtered': {'total': filt_total,
                     'by_vid': {v: len(by_vid_filtered.get(v,[])) for v in VID_ORDER},
                     'by_split': {s: len(split_frames[s]) for s in ('train','val','test')}},
        'stats': {
            'raw':      raw_stats,
            'filtered': filt_stats,
        },
        'split_detail': stats_by_split,
        'quality': {'issues_total': issues_total},
    }
    rp = DST / 'reports' / 'build_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 콘솔 최종 출력 ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print("  📊 dataset_field_A_v1 구축 완료")
    print(f"{'='*65}\n")

    print("① 원본 CASE A frame 수 (stride 적용 전)")
    for vid_id in VID_ORDER:
        n = len(by_vid_raw.get(vid_id, []))
        print(f"   {vid_id}[{VIDS_INFO[vid_id]}]: {n}장")
    print(f"   합계: {raw_total}장  /  monster bbox: {raw_stats['bbox_total']}개")

    print(f"\n② stride 적용 후 (min_gap={MIN_GAP})")
    for vid_id in VID_ORDER:
        raw_n  = len(by_vid_raw.get(vid_id, []))
        filt_n = len(by_vid_filtered.get(vid_id, []))
        print(f"   {vid_id}[{VIDS_INFO[vid_id]}]: {raw_n} → {filt_n}장  "
              f"(-{raw_n-filt_n})")
    print(f"   합계: {raw_total} → {filt_total}장  "
          f"({filt_total/raw_total*100:.1f}% 유지)")

    print(f"\n③ train/val/test 분포")
    for sp in ('train', 'val', 'test'):
        sd = stats_by_split[sp]
        print(f"   {sp:5s}: {sd['images']:3d}이미지  "
              f"bbox={sd['bbox']}  empty_label={sd['empty']}")
    total_bbox = sum(s['bbox'] for s in stats_by_split.values())
    print(f"   합계 bbox: {total_bbox}개")

    print(f"\n④ 위치 분포 (stride 후, {filt_stats['bbox_total']}개 기준)")
    pos = filt_stats['position']
    tot_pos = sum(pos.values()) or 1
    for tag in ('LEFT','CENTER','RIGHT','TOP','MIDDLE','BOTTOM'):
        n   = pos.get(tag, 0)
        pct = n / tot_pos * 100
        bar = '█' * int(pct/4+0.5)
        status = '✅' if pct >= 10 else ('⚠️' if pct >= 4 else '❌')
        print(f"   {tag:8s}: {n:4d} ({pct:5.1f}%)  {bar:<12s} {status}")

    print(f"\n⑤ 크기 분포")
    sz = filt_stats['size']
    tot_sz = sum(sz.values()) or 1
    for tag in ('SMALL','MEDIUM','LARGE'):
        n = sz.get(tag,0)
        print(f"   {tag:6s}: {n:4d} ({n/tot_sz*100:.1f}%)")

    print(f"\n⑥ 연속 프레임")
    print(f"   stride 전 최대 연속: {raw_stats['max_consecutive']}프레임")
    print(f"   stride 후 최대 연속: {filt_stats['max_consecutive']}프레임")

    print(f"\n⑦ 라벨 품질: issues={issues_total}")

    print(f"\n⑧ Contact Sheet 경로")
    for p in sorted((DST/'contact_sheets').glob('*.jpg')):
        print(f"   {p.relative_to(BASE)}")

    print(f"\n⑨ data.yaml: {DST/'data.yaml'}")
    print(f"   dataset_field_v1 보존: ✅ (B/C 라벨 삭제 없음)")

    print(f"\n  소요: {elapsed:.1f}초")
    print(f"{'='*65}")
    print("  ✅ 학습 미실행. 데이터셋 확인 후 승인하면 진행합니다.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
