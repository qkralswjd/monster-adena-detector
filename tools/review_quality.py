"""
tools/review_quality.py
=======================
라벨 품질 검수용 고품질 Preview + Contact Sheet 생성기.

- CASE B (미탐 → 새 GT): 영상 6개에서 균형 있게 샘플링, 60장 내외
- CASE C (저신뢰 트랙 보완): 영상 6개에서 균형 있게 샘플링, 40장 내외
- 각 preview: 풀 ROI 화면 + GT bbox(두꺼운 색) + YOLO bbox(점선) + 정보 텍스트
- Contact sheet: 한눈에 비교 가능하도록 grid 배열

출력 디렉토리: dataset_field_v1/review/
  review/B_previews/  - B case 개별 preview
  review/C_previews/  - C case 개별 preview
  review/A_previews/  - A case 비교용 샘플
  review/sheet_B_01.jpg ~ sheet_B_04.jpg  - B contact sheet (페이지 분할)
  review/sheet_C_01.jpg ~ sheet_C_02.jpg  - C contact sheet
  review/sheet_A_01.jpg                   - A 비교용 contact sheet

사용법:
  python tools/review_quality.py
"""

from __future__ import annotations
import cv2, json, math, random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATASET = BASE / 'dataset_field_v1'
REVIEW  = DATASET / 'review'

# ─── ROI 좌표 ────────────────────────────────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 372, 259, 1135, 472

# ─── 색상 정의 (BGR) ─────────────────────────────────────────────────────────
COLOR_GT   = (0,   220,  0)     # 초록 - Ground Truth bbox
COLOR_YOLO = (0,   200, 255)    # 노란색 - YOLO bbox (기존)
COLOR_MISS = (0,    80, 255)    # 주황 - YOLO 미탐 표시
COLOR_LOW  = (255, 140,   0)    # 파랑 - 저신뢰
COLOR_BG   = (30,   30,  30)    # 텍스트 배경
COLOR_INFO = (220, 220, 220)    # 정보 텍스트

# ─── Preview 크기 ────────────────────────────────────────────────────────────
PREV_W      = 1135   # ROI 원본 폭 그대로 사용
PREV_H      = 472    # ROI 원본 높이 그대로 사용
THUMB_W     = 540    # contact sheet 썸네일 폭
THUMB_H     = 224    # 썸네일 높이 (ROI 비율 유지)
SHEET_COLS  = 3      # contact sheet 열 수
INFO_H      = 44     # 하단 정보 바 높이

# ─── 영상별 목표 샘플 수 (B: 총 ~60장, C: 총 ~40장) ─────────────────────────
VID_ORDER = ['VID_A','VID_B','VID_C','VID_D','VID_E','VID_F']
VIDS_INFO = {
    'VID_A': 'train', 'VID_B': 'train', 'VID_C': 'train',
    'VID_D': 'val',   'VID_E': 'val',   'VID_F': 'test',
}


def load_all_results() -> Dict[str, List[Dict]]:
    """모든 영상의 gt_results 캐시를 로드. {case: [frame_record, ...]}"""
    all_by_case: Dict[str, List[Dict]] = defaultdict(list)

    for vid_id in VID_ORDER:
        cache = DATASET / 'reports' / f'{vid_id}_gt_results.json'
        if not cache.exists():
            print(f"  [WARN] 캐시 없음: {cache}")
            continue
        with open(cache) as f:
            raw = json.load(f)
        for k, fr in raw.items():
            fr['vid_id']  = vid_id
            fr['split']   = VIDS_INFO[vid_id]
            fr['frame_no'] = int(k)
            all_by_case[fr['case']].append(fr)

    return all_by_case


def sample_balanced(frames: List[Dict], target: int, seed: int = 42) -> List[Dict]:
    """영상별로 균형 있게 샘플링. 각 영상에서 비례 배분."""
    rng = random.Random(seed)

    by_vid: Dict[str, List[Dict]] = defaultdict(list)
    for fr in frames:
        by_vid[fr['vid_id']].append(fr)

    # 각 영상별 프레임을 frame_no 순 정렬 후 균등 간격 선택
    result = []
    total = len(frames)
    for vid_id in VID_ORDER:
        vframes = sorted(by_vid.get(vid_id, []), key=lambda x: x['frame_no'])
        if not vframes:
            continue
        n_vid = max(1, round(len(vframes) / total * target))
        if len(vframes) <= n_vid:
            result.extend(vframes)
        else:
            # 균등 간격으로 선택 (연속 프레임 방지)
            step = len(vframes) / n_vid
            selected = [vframes[int(i * step)] for i in range(n_vid)]
            result.extend(selected)

    # target 초과 시 trim
    if len(result) > target:
        result = sorted(result, key=lambda x: x['frame_no'])
        # 균등하게 줄이기
        step = len(result) / target
        result = [result[int(i * step)] for i in range(target)]

    return sorted(result, key=lambda x: (x['vid_id'], x['frame_no']))


def draw_review_preview(fr: Dict, out_path: Path,
                        full_frame_path: Optional[Path] = None) -> bool:
    """
    고품질 검수용 Preview 생성.
    - ROI 크롭 이미지 (원본 해상도)
    - GT bbox: 두꺼운 초록색 실선 + 라벨
    - YOLO bbox: 하늘색 점선 (있는 경우)
    - 하단 정보 바: frame번호, 타임스탬프, case, conf, track_len
    - CASE B: "YOLO MISS" 배지
    - CASE C: "LOW CONF" + 실제 conf 값
    """
    # 원본 프레임 로드
    img_path = Path(fr['path'])
    if not img_path.exists():
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        return False

    H, W = img.shape[:2]

    # ROI 크롭
    rx1 = max(0, ROI_X)
    ry1 = max(0, ROI_Y)
    rx2 = min(W, ROI_X + ROI_W)
    ry2 = min(H, ROI_Y + ROI_H)
    roi = img[ry1:ry2, rx1:rx2].copy()
    rh, rw = roi.shape[:2]

    # ── YOLO bbox 그리기 (점선 하늘색) ──────────────────────────────────────
    for det in fr.get('yolo_dets', []):
        conf = det.get('conf', 0)
        x1 = int(det['x1r'])
        y1 = int(det['y1r'])
        x2 = int(det['x2r'])
        y2 = int(det['y2r'])
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(rw-1, x2), min(rh-1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        # 점선 효과: 세그먼트 단위로 그리기
        _draw_dashed_rect(roi, (x1c, y1c, x2c, y2c), COLOR_YOLO, thickness=2, dash=8)
        label = f"Y:{conf:.2f}"
        cv2.putText(roi, label, (x1c+2, y1c-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YOLO, 1, cv2.LINE_AA)

    # ── GT bbox 그리기 (굵은 초록 실선) ────────────────────────────────────
    case = fr.get('case', '?')
    gt_color = {
        'A': (0, 220, 0),      # 초록
        'B': (0, 100, 255),    # 주황 (YOLO 미탐 발굴)
        'C': (0, 180, 255),    # 노랑 (저신뢰 보완)
        'E': (0, 0, 220),      # 빨강 (오탐)
        'HARD': (200, 0, 200), # 보라 (애매)
    }.get(case, (0, 220, 0))

    for det in fr.get('gt_dets', []):
        x1 = int(det['x1r'])
        y1 = int(det['y1r'])
        x2 = int(det['x2r'])
        y2 = int(det['y2r'])
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(rw-1, x2), min(rh-1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        conf      = det.get('conf', 0)
        track_len = det.get('track_len', 0)
        gt_source = det.get('gt_source', '')
        clipped   = det.get('clipped', False)

        # 메인 GT bbox
        cv2.rectangle(roi, (x1c, y1c), (x2c, y2c), gt_color, thickness=3)

        # 잘린 경우 점선 테두리 추가
        if clipped:
            cv2.rectangle(roi, (x1c-2, y1c-2), (x2c+2, y2c+2),
                          (0, 0, 255), thickness=1)

        # 라벨 구성
        cls_name = det.get('cls_name', 'obj')
        label_top = f"{cls_name} {conf:.2f}"
        label_bot = f"tr:{track_len}"
        if gt_source == 'track_interpolated':
            label_bot += " interp"

        # 라벨 배경
        lw = max(cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0],
                 cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0])
        lx1, ly1 = x1c, max(0, y1c - 30)
        cv2.rectangle(roi, (lx1, ly1), (lx1 + lw + 4, y1c), (0, 0, 0), -1)
        cv2.putText(roi, label_top, (lx1+2, y1c-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1, cv2.LINE_AA)
        cv2.putText(roi, label_bot, (lx1+2, y1c-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # ── CASE 배지 ────────────────────────────────────────────────────────────
    badge_text = {
        'B': 'YOLO MISS',
        'C': 'LOW CONF',
        'E': 'FALSE POS',
        'HARD': 'HARD',
        'A': 'OK',
        'EMPTY': 'EMPTY',
    }.get(case, case)
    badge_color = {
        'B': (0, 80, 255),
        'C': (0, 160, 255),
        'E': (0, 0, 200),
        'HARD': (180, 0, 180),
        'A': (0, 180, 0),
        'EMPTY': (100, 100, 100),
    }.get(case, (128, 128, 128))

    bw, bh = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.rectangle(roi, (rw - bw - 14, 4), (rw - 4, bh + 12), badge_color, -1)
    cv2.putText(roi, badge_text, (rw - bw - 10, bh + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # ── 하단 정보 바 ─────────────────────────────────────────────────────────
    info_bar = np.zeros((INFO_H, rw, 3), dtype=np.uint8)
    info_bar[:] = (25, 25, 25)

    vid_id = fr.get('vid_id', '?')
    split  = fr.get('split', '?')
    fn     = fr.get('frame_no', 0)
    ts     = fr.get('ts_sec', 0.0)
    diff   = fr.get('diff', 0.0)
    n_gt   = len(fr.get('gt_dets', []))
    n_yolo = len(fr.get('yolo_dets', []))

    info1 = f"{vid_id}[{split}] f{fn:06d} t={ts:.1f}s  diff={diff:.1f}"
    info2 = f"CASE={case}  GT={n_gt}box  YOLO={n_yolo}box"

    cv2.putText(info_bar, info1, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(info_bar, info2, (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, gt_color, 1, cv2.LINE_AA)

    # 합치기
    combined = np.vstack([roi, info_bar])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def _draw_dashed_rect(img, rect, color, thickness=1, dash=8):
    """점선 사각형 그리기."""
    x1, y1, x2, y2 = rect
    def dashed_line(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        n_segs = max(1, int(length / dash))
        for i in range(n_segs):
            if i % 2 == 1:
                continue
            t0 = i / n_segs
            t1 = min(1.0, (i + 0.8) / n_segs)
            sx = int(p1[0] + dx * t0)
            sy = int(p1[1] + dy * t0)
            ex = int(p1[0] + dx * t1)
            ey = int(p1[1] + dy * t1)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def make_contact_sheet_paged(prev_paths: List[Path], out_prefix: Path,
                              title: str, cols: int = SHEET_COLS,
                              page_size: int = 24,
                              thumb_w: int = THUMB_W,
                              thumb_h: int = THUMB_H) -> List[Path]:
    """
    여러 preview를 page_size장씩 나눠 contact sheet 저장.
    각 썸네일: thumb_w × (thumb_h + INFO_H) → 이미 info_bar 포함된 이미지 사용.
    """
    pages = []
    total = len(prev_paths)
    n_pages = math.ceil(total / page_size)

    for pg in range(n_pages):
        chunk = prev_paths[pg * page_size: (pg + 1) * page_size]
        rows  = math.ceil(len(chunk) / cols)

        sheet_w = cols * thumb_w
        sheet_h = rows * (thumb_h + INFO_H) + 50  # 상단 타이틀 영역
        sheet   = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        sheet[:] = (20, 20, 20)

        # 타이틀
        page_title = f"{title} (Page {pg+1}/{n_pages})"
        cv2.putText(sheet, page_title, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

        for idx, p in enumerate(chunk):
            img = cv2.imread(str(p))
            if img is None:
                continue
            # 썸네일 리사이즈 (info_bar 포함된 전체 이미지)
            thumb = cv2.resize(img, (thumb_w, thumb_h + INFO_H))

            row = idx // cols
            col = idx % cols
            y0  = 50 + row * (thumb_h + INFO_H)
            x0  = col * thumb_w

            sheet[y0:y0 + thumb_h + INFO_H, x0:x0 + thumb_w] = thumb

            # 경계선
            cv2.rectangle(sheet, (x0, y0),
                          (x0 + thumb_w - 1, y0 + thumb_h + INFO_H - 1),
                          (80, 80, 80), 1)

        out_path = Path(str(out_prefix) + f'_{pg+1:02d}.jpg')
        cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pages.append(out_path)
        print(f"  → {out_path.name} ({len(chunk)}장)")

    return pages


def calc_consecutive_stats(frames: List[Dict]) -> Dict:
    """
    연속 프레임 통계 계산.
    같은 영상 내에서 frame_no 차이가 ≤6인 연속 그룹을 찾아 분석.
    """
    # 영상별 정렬
    by_vid: Dict[str, List[int]] = defaultdict(list)
    for fr in frames:
        by_vid[fr['vid_id']].append(fr['frame_no'])

    result = {}
    total_consec = 0
    for vid_id in VID_ORDER:
        fnums = sorted(by_vid.get(vid_id, []))
        if not fnums:
            continue
        groups = []
        cur_group = [fnums[0]]
        for fn in fnums[1:]:
            if fn - cur_group[-1] <= 6:  # 6프레임(=EXTRACT_INTERVAL*2) 이하면 연속
                cur_group.append(fn)
            else:
                groups.append(cur_group)
                cur_group = [fn]
        groups.append(cur_group)

        long_groups = [g for g in groups if len(g) >= 4]  # 4장 이상 연속
        consec_frames = sum(len(g) for g in long_groups)
        total_consec += consec_frames
        result[vid_id] = {
            'total': len(fnums),
            'groups': len(groups),
            'long_groups (>=4)': len(long_groups),
            'frames_in_long_groups': consec_frames,
            'max_group_len': max(len(g) for g in groups),
        }

    result['total_consecutive_risk'] = total_consec
    return result


def main():
    print("=" * 65)
    print("  📋 라벨 품질 검수용 Preview + Contact Sheet 생성")
    print("=" * 65)

    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / 'B_previews').mkdir(exist_ok=True)
    (REVIEW / 'C_previews').mkdir(exist_ok=True)
    (REVIEW / 'A_previews').mkdir(exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n1. GT 결과 캐시 로드...")
    all_by_case = load_all_results()

    for case in ['A', 'B', 'C', 'E', 'HARD', 'EMPTY']:
        n = len(all_by_case.get(case, []))
        print(f"   CASE {case:5s}: {n:4d}개")

    # ── 연속 프레임 통계 (전체 GT 기준) ─────────────────────────────────────
    print("\n2. 연속 프레임 통계 분석...")
    all_frames = []
    for lst in all_by_case.values():
        all_frames.extend(lst)

    # CASE 별 연속 통계
    print("\n   [전체 GT 연속 프레임 분석 - 영상별]")
    consec_stats = calc_consecutive_stats(all_frames)
    for vid_id in VID_ORDER:
        s = consec_stats.get(vid_id, {})
        if not s:
            continue
        print(f"   {vid_id}: 총{s['total']}장, "
              f"그룹{s['groups']}개, "
              f"장기연속(≥4){s['long_groups (>=4)']}그룹/{s['frames_in_long_groups']}장, "
              f"최대{s['max_group_len']}연속")
    print(f"   ⚠ 장기연속 위험 프레임 합계: {consec_stats['total_consecutive_risk']}장")

    # B case 연속 통계
    print("\n   [CASE B 연속 프레임 분석]")
    b_consec = calc_consecutive_stats(all_by_case.get('B', []))
    for vid_id in VID_ORDER:
        s = b_consec.get(vid_id, {})
        if not s:
            continue
        print(f"   {vid_id}: 총{s['total']}장, 장기연속{s['frames_in_long_groups']}장, 최대{s['max_group_len']}연속")

    # ── CASE B Preview 생성 ──────────────────────────────────────────────────
    print("\n3. CASE B Preview 생성 (영상별 균형 샘플링)...")
    b_frames = all_by_case.get('B', [])

    # 전략: 영상별로 최소 5장 + 나머지 비례 배분 → 총 60장
    b_sample = sample_balanced(b_frames, target=60, seed=42)
    print(f"   선택: {len(b_sample)}장 / 전체 {len(b_frames)}장")

    # 영상별 분포 확인
    b_vid_dist: Dict[str, int] = defaultdict(int)
    for fr in b_sample:
        b_vid_dist[fr['vid_id']] += 1
    for vid_id in VID_ORDER:
        print(f"   {vid_id}: {b_vid_dist.get(vid_id, 0)}장")

    b_prev_paths = []
    for i, fr in enumerate(b_sample):
        out_p = REVIEW / 'B_previews' / f"B_{fr['vid_id']}_f{fr['frame_no']:06d}.jpg"
        ok = draw_review_preview(fr, out_p)
        if ok:
            b_prev_paths.append(out_p)
        if (i + 1) % 20 == 0:
            print(f"   B preview: {i+1}/{len(b_sample)}")

    print(f"   B preview 완료: {len(b_prev_paths)}장")

    # ── CASE C Preview 생성 ──────────────────────────────────────────────────
    print("\n4. CASE C Preview 생성 (영상별 균형 샘플링)...")
    c_frames = all_by_case.get('C', [])
    c_sample = sample_balanced(c_frames, target=42, seed=42)
    print(f"   선택: {len(c_sample)}장 / 전체 {len(c_frames)}장")

    c_vid_dist: Dict[str, int] = defaultdict(int)
    for fr in c_sample:
        c_vid_dist[fr['vid_id']] += 1
    for vid_id in VID_ORDER:
        print(f"   {vid_id}: {c_vid_dist.get(vid_id, 0)}장")

    c_prev_paths = []
    for i, fr in enumerate(c_sample):
        out_p = REVIEW / 'C_previews' / f"C_{fr['vid_id']}_f{fr['frame_no']:06d}.jpg"
        ok = draw_review_preview(fr, out_p)
        if ok:
            c_prev_paths.append(out_p)
        if (i + 1) % 20 == 0:
            print(f"   C preview: {i+1}/{len(c_sample)}")

    print(f"   C preview 완료: {len(c_prev_paths)}장")

    # ── CASE A Preview 생성 (비교용, 20장) ───────────────────────────────────
    print("\n5. CASE A Preview 생성 (비교 기준, 20장)...")
    a_frames = all_by_case.get('A', [])
    a_sample = sample_balanced(a_frames, target=20, seed=42)

    a_prev_paths = []
    for fr in a_sample:
        out_p = REVIEW / 'A_previews' / f"A_{fr['vid_id']}_f{fr['frame_no']:06d}.jpg"
        ok = draw_review_preview(fr, out_p)
        if ok:
            a_prev_paths.append(out_p)
    print(f"   A preview 완료: {len(a_prev_paths)}장")

    # ── Contact Sheet 생성 ───────────────────────────────────────────────────
    print("\n6. Contact Sheet 생성...")

    # B: 4페이지 (60장 → 24장씩 → 3페이지)
    print("   CASE B contact sheet...")
    b_sheets = make_contact_sheet_paged(
        b_prev_paths, REVIEW / 'sheet_B',
        "CASE B: YOLO 미탐 → 새 GT (몬스터 존재하는데 YOLO가 놓침)",
        cols=SHEET_COLS, page_size=18)

    # C: 2페이지 (42장 → 21장씩 → 2페이지)
    print("   CASE C contact sheet...")
    c_sheets = make_contact_sheet_paged(
        c_prev_paths, REVIEW / 'sheet_C',
        "CASE C: 저신뢰 YOLO + 트랙 연속성으로 보완",
        cols=SHEET_COLS, page_size=18)

    # A: 1페이지 (비교 기준)
    print("   CASE A contact sheet...")
    a_sheets = make_contact_sheet_paged(
        a_prev_paths, REVIEW / 'sheet_A',
        "CASE A: 고신뢰 탐지 성공 (비교 기준)",
        cols=SHEET_COLS, page_size=18)

    # ── 최종 수치 보고 ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  📊 최종 수치 보고")
    print("=" * 65)

    # 전체 GT frame 수
    labeled_cases = ['A', 'B', 'C']
    total_gt_frames = sum(len(all_by_case.get(c, [])) for c in labeled_cases)
    total_all_frames = sum(len(v) for v in all_by_case.values())

    print(f"\n① 전체 GT frame 수")
    print(f"   라벨 있는 프레임 (A+B+C): {total_gt_frames}장")
    print(f"   전체 프레임 (E+EMPTY+HARD 포함): {total_all_frames}장")

    print(f"\n② CASE별 개수")
    for case in ['A', 'B', 'C', 'E', 'HARD', 'EMPTY']:
        n = len(all_by_case.get(case, []))
        ratio = n / total_all_frames * 100 if total_all_frames else 0
        print(f"   {case:5s}: {n:4d}장 ({ratio:5.1f}%)")

    print(f"\n③ 실제 monster bbox 개수")
    bbox_count = {'A': 0, 'B': 0, 'C': 0}
    for case in ['A', 'B', 'C']:
        for fr in all_by_case.get(case, []):
            for det in fr.get('gt_dets', []):
                if det.get('cls_name') == 'monster':
                    bbox_count[case] += 1
    total_bbox = sum(bbox_count.values())
    for case, cnt in bbox_count.items():
        print(f"   CASE {case}: {cnt}개")
    print(f"   합계: {total_bbox}개")

    print(f"\n④ 영상별 GT 개수 (A+B+C 라벨 프레임)")
    for vid_id in VID_ORDER:
        by_vid_case: Dict[str, int] = defaultdict(int)
        for case in ['A', 'B', 'C']:
            for fr in all_by_case.get(case, []):
                if fr['vid_id'] == vid_id:
                    by_vid_case[case] += 1
        total_v = sum(by_vid_case.values())
        split = VIDS_INFO[vid_id]
        detail = ' '.join(f"{c}={by_vid_case[c]}" for c in ['A','B','C'])
        print(f"   {vid_id}[{split}]: {total_v}장  ({detail})")

    print(f"\n⑤ train/val/test 개수")
    split_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in ['A', 'B', 'C']:
        for fr in all_by_case.get(case, []):
            split_count[fr['split']][case] += 1
    for split in ['train', 'val', 'test']:
        sc = split_count[split]
        total_s = sum(sc.values())
        detail = ' '.join(f"{c}={sc.get(c,0)}" for c in ['A','B','C'])
        print(f"   {split:5s}: {total_s}장  ({detail})")

    print(f"\n⑥ 연속/중복 프레임 위험")
    print(f"   장기연속(≥4프레임 연속) 위험: {consec_stats['total_consecutive_risk']}장")
    for vid_id in VID_ORDER:
        s = consec_stats.get(vid_id, {})
        if s:
            print(f"   {vid_id}: 최대 {s['max_group_len']}연속, "
                  f"위험 {s['frames_in_long_groups']}장")

    print(f"\n⑦ Hard case 개수")
    print(f"   HARD(애매): {len(all_by_case.get('HARD', []))}장")
    print(f"   E(오탐FP):  {len(all_by_case.get('E', []))}장")
    print(f"   EMPTY(빈):  {len(all_by_case.get('EMPTY', []))}장")

    print(f"\n⑧ Preview / Contact Sheet 경로")
    print(f"   review/B_previews/  ({len(b_prev_paths)}장)")
    print(f"   review/C_previews/  ({len(c_prev_paths)}장)")
    print(f"   review/A_previews/  ({len(a_prev_paths)}장, 비교 기준)")
    all_sheets = b_sheets + c_sheets + a_sheets
    for sp in all_sheets:
        print(f"   {sp.relative_to(DATASET)}")

    print(f"\n{'=' * 65}")
    print("  ✅ 검수 준비 완료. Contact sheet를 확인하고 승인 후 학습 진행.")
    print(f"{'=' * 65}\n")

    return {
        'b_sheets': b_sheets,
        'c_sheets': c_sheets,
        'a_sheets': a_sheets,
        'b_prev': b_prev_paths,
        'c_prev': c_prev_paths,
        'a_prev': a_prev_paths,
        'consec_stats': consec_stats,
    }


if __name__ == '__main__':
    main()
