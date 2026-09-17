"""
analyze_detection_session.py
============================
detection_validator.py 가 저장한 세션 데이터를 분석해
10개 섹션 최종 보고서를 출력한다.

사용법:
  python analyze_detection_session.py <세션_디렉토리>
  python analyze_detection_session.py detection_sessions/2026-09-17_10-00-00

  # 최신 세션 자동 선택
  python analyze_detection_session.py

출력:
  ① 실제 몬스터 탐지율
  ② 연속 탐지 안정성
  ③ confidence 분포
  ④ 위치별 탐지율
  ⑤ 몬스터 크기별 탐지율
  ⑥ 배경별 탐지율 (추론)
  ⑦ 이동 중 탐지율
  ⑧ 가장 많이 발생하는 미탐 원인
  ⑨ 현재 학습 데이터에서 부족한 부분
  ⑩ 탐지 성능을 높이기 위해 가장 먼저 해야 할 작업
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ── 위치 기준 (detection_validator.py 와 동일) ───────────────────────────────
ZONE_LEFT_EDGE  = 120
ZONE_RIGHT_EDGE = 1015
ZONE_BTM_EDGE   = 380
ZONE_TOP_EDGE   = 80
SIZE_SMALL  = 40
SIZE_LARGE  = 100

# 학습 데이터 실측값 (이전 세션 분석 결과)
TRAIN_STATS = {
    'total_monster': 621,
    'cy_0_200_pct':   58.1,
    'cy_200_300_pct': 25.0,
    'cy_300_380_pct': 11.1,
    'cy_380_472_pct':  0.0,   # ← 완전 공백
    'cx_0_100_pct':    0.0,   # ← 완전 공백
    'cx_1000_1135_pct': 0.0,  # ← 완전 공백
    'avg_w': 52.7,
    'avg_h': 45.8,
    'min_w': 19.5,
    'min_h': 20.1,
}


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_session(sess_dir: str):
    log_path    = os.path.join(sess_dir, 'frame_log.json')
    tracks_path = os.path.join(sess_dir, 'tracks.json')

    if not os.path.exists(log_path):
        print(f"[ERROR] frame_log.json 없음: {log_path}")
        sys.exit(1)

    with open(log_path, encoding='utf-8') as f:
        frames = json.load(f)

    tracks = []
    if os.path.exists(tracks_path):
        with open(tracks_path, encoding='utf-8') as f:
            tracks = json.load(f)

    return frames, tracks


def latest_session(base: str = 'detection_sessions') -> str:
    dirs = sorted(glob.glob(os.path.join(base, '*')))
    if not dirs:
        print(f"[ERROR] 세션 없음: {base}")
        sys.exit(1)
    return dirs[-1]


# ══════════════════════════════════════════════════════════════════════════════
# 섹션별 분석 함수
# ══════════════════════════════════════════════════════════════════════════════

def sec01_overall(frames: List, tracks: List) -> Dict:
    """① 실제 몬스터 탐지율"""
    total_frames = len(frames)
    det_frames   = sum(1 for f in frames if f['n_monster'] > 0)
    all_dets     = [d for f in frames for d in f['dets']]
    det_frames_20 = sum(1 for f in frames
                        if any(d['conf'] >= 0.20 for d in f['dets']))
    det_frames_15 = sum(1 for f in frames
                        if any(d['conf'] >= 0.15 for d in f['dets']))

    return {
        'total_frames':    total_frames,
        'det_frames_any':  det_frames,
        'det_rate_any':    det_frames / total_frames if total_frames else 0,
        'det_frames_20':   det_frames_20,
        'det_rate_20':     det_frames_20 / total_frames if total_frames else 0,
        'det_frames_15':   det_frames_15,
        'det_rate_15':     det_frames_15 / total_frames if total_frames else 0,
        'total_dets':      len(all_dets),
        'miss_frames':     total_frames - det_frames,
    }


def sec02_stability(tracks: List) -> Dict:
    """② 연속 탐지 안정성"""
    if not tracks:
        return {'tracks': 0}
    recalls   = [t['recall'] for t in tracks]
    max_gaps  = [t['max_gap'] for t in tracks]
    stable    = sum(1 for t in tracks if t['recall'] >= 0.8)
    unstable  = sum(1 for t in tracks if t['recall'] < 0.5)

    return {
        'total_tracks':  len(tracks),
        'avg_recall':    float(np.mean(recalls)),
        'min_recall':    float(np.min(recalls)),
        'max_gap_frames': int(np.max(max_gaps)),
        'avg_gap':       float(np.mean(max_gaps)),
        'stable_tracks': stable,    # recall >= 80%
        'unstable_tracks': unstable, # recall < 50%
    }


def sec03_confidence(frames: List) -> Dict:
    """③ confidence 분포"""
    all_dets = [d for f in frames for d in f['dets']]
    if not all_dets:
        return {'count': 0}
    confs = [d['conf'] for d in all_dets]
    bins  = [
        (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
        (0.20, 0.25), (0.25, 0.30), (0.30, 0.50), (0.50, 1.01),
    ]
    hist = {}
    for lo, hi in bins:
        cnt = sum(1 for c in confs if lo <= c < hi)
        hist[f'{lo:.2f}-{hi:.2f}'] = cnt
    return {
        'count':   len(confs),
        'avg':     float(np.mean(confs)),
        'std':     float(np.std(confs)),
        'min':     float(np.min(confs)),
        'max':     float(np.max(confs)),
        'pct_gte_20': sum(1 for c in confs if c >= 0.20) / len(confs),
        'pct_gte_15': sum(1 for c in confs if c >= 0.15) / len(confs),
        'histogram': hist,
        'unstable_15_25': sum(1 for c in confs if 0.15 <= c < 0.25) / len(confs),
    }


def sec04_zone(frames: List) -> Dict:
    """④ 위치별 탐지율"""
    all_dets = [d for f in frames for d in f['dets']]
    if not all_dets:
        return {}

    zones = {}
    for d in all_dets:
        z = d['zone']
        if z not in zones:
            zones[z] = {'count': 0, 'confs': []}
        zones[z]['count'] += 1
        zones[z]['confs'].append(d['conf'])

    result = {}
    total = len(all_dets)
    for z, v in zones.items():
        result[z] = {
            'count':    v['count'],
            'pct':      v['count'] / total,
            'avg_conf': float(np.mean(v['confs'])),
        }

    # 경계 탐지 비율
    edge_cnt = sum(1 for d in all_dets
                   if d['cx'] < ZONE_LEFT_EDGE or d['cx'] > ZONE_RIGHT_EDGE
                   or d['cy'] > ZONE_BTM_EDGE)
    btm_cnt  = sum(1 for d in all_dets if d['cy'] > ZONE_BTM_EDGE)
    result['_edge_pct']  = edge_cnt / total
    result['_btm_pct']   = btm_cnt  / total
    result['_clip_count'] = sum(1 for d in all_dets if d['clip'])

    return result


def sec05_size(frames: List) -> Dict:
    """⑤ 몬스터 크기별 탐지율"""
    all_dets = [d for f in frames for d in f['dets']]
    if not all_dets:
        return {}

    sizes = {}
    for d in all_dets:
        s = d['size']
        if s not in sizes:
            sizes[s] = {'count': 0, 'confs': [], 'areas': []}
        sizes[s]['count'] += 1
        sizes[s]['confs'].append(d['conf'])
        sizes[s]['areas'].append(d['w'] * d['h'])

    total = len(all_dets)
    ws = [d['w'] for d in all_dets]
    hs = [d['h'] for d in all_dets]
    result = {
        'avg_w': float(np.mean(ws)),
        'avg_h': float(np.mean(hs)),
        'min_w': float(np.min(ws)),
        'min_h': float(np.min(hs)),
    }
    for s, v in sizes.items():
        result[s] = {
            'count':    v['count'],
            'pct':      v['count'] / total,
            'avg_conf': float(np.mean(v['confs'])),
            'avg_area': float(np.mean(v['areas'])),
        }
    return result


def sec06_background(frames: List) -> Dict:
    """⑥ 배경별 탐지율 (inference — 영상 직접 확인 필요)"""
    # detection_validator는 배경 정보를 직접 수집하지 않음.
    # 실제 영상 확인 후 수동으로 보완 필요.
    # 여기서는 cy 구간을 배경 프록시로 사용:
    #   cy < 150 → 원경 / 하늘 배경 (상단)
    #   cy 150~300 → 중경 배경
    #   cy > 300 → 근경 / 지면 배경 (하단)
    all_dets = [d for f in frames for d in f['dets']]
    if not all_dets:
        return {'note': '탐지 없음 — 실제 영상 수동 확인 필요'}

    bg_proxy = {
        '상단(cy<150)':   [d for d in all_dets if d['cy'] < 150],
        '중단(150-300)':  [d for d in all_dets if 150 <= d['cy'] < 300],
        '하단(300-380)':  [d for d in all_dets if 300 <= d['cy'] < 380],
        '최하단(cy>380)': [d for d in all_dets if d['cy'] >= 380],
    }
    total = len(all_dets)
    result = {'note': 'cy 구간 기반 추론 — 실제 배경은 영상 수동 확인 필요'}
    for label, dets in bg_proxy.items():
        if dets:
            result[label] = {
                'count':    len(dets),
                'pct':      len(dets) / total,
                'avg_conf': float(np.mean([d['conf'] for d in dets])),
            }
        else:
            result[label] = {'count': 0, 'pct': 0.0, 'avg_conf': 0.0}
    return result


def sec07_movement(frames: List, tracks: List) -> Dict:
    """⑦ 이동 중 탐지율 — 연속 프레임 간 cx+cy 변화량으로 이동 판단"""
    moving_det   = 0   # 이동 중 탐지
    stationary_det = 0 # 정지 중 탐지
    moving_miss  = 0   # 이동 중 미탐 (직전 탐지→현재 미탐)
    MOVE_THRESH  = 5   # px (cx²+cy²)^0.5 이상이면 '이동 중' (10fps 기준 0.5px/frame~게임 이동 충분)

    for i in range(1, len(frames)):
        f_cur  = frames[i]
        f_prev = frames[i - 1]

        # 직전 프레임에 탐지 있어야 이동 추론 가능
        if not f_prev['dets']:
            continue

        if f_cur['dets']:
            # 현재도 탐지 → 가장 가까운 쌍으로 이동량 측정
            for d_cur in f_cur['dets']:
                best_dist = 9999
                for d_prev in f_prev['dets']:
                    dist2d = ((d_cur['cx'] - d_prev['cx']) ** 2 +
                              (d_cur['cy'] - d_prev['cy']) ** 2) ** 0.5
                    if dist2d < best_dist:
                        best_dist = dist2d
                if best_dist < 80:   # 같은 몬스터로 간주
                    if best_dist >= MOVE_THRESH:
                        moving_det += 1
                    else:
                        stationary_det += 1
        else:
            # 현재 미탐 → 직전 탐지 cx/cy로 이동 방향 추론 불가,
            # 직전→직전전 비교로 추론
            if i >= 2 and frames[i - 2]['dets']:
                for d1 in f_prev['dets']:
                    for d0 in frames[i - 2]['dets']:
                        dist2d = ((d1['cx'] - d0['cx']) ** 2 +
                                  (d1['cy'] - d0['cy']) ** 2) ** 0.5
                        if dist2d >= MOVE_THRESH:
                            moving_miss += 1
                            break
                    break  # 첫 탐지만

    total_det = moving_det + stationary_det
    return {
        'moving_det':        moving_det,
        'stationary_det':    stationary_det,
        'moving_miss':       moving_miss,
        'det_rate_moving':   moving_det / (moving_det + moving_miss) if (moving_det + moving_miss) else 0,
        'det_rate_stationary': stationary_det / total_det if total_det else 0,
        'note': f'이동 기준: 연속 프레임 간 거리 >= {MOVE_THRESH}px  (10fps 기준, 플레이어/몬스터 이동 모두 포함)',
    }


def sec08_miss_causes(frames: List, tracks: List) -> Dict:
    """⑧ 미탐 원인 분류"""
    all_dets = [d for f in frames for d in f['dets']]

    # 경계 clipping 발생 (탐지는 됐지만 절반이 잘린 케이스)
    clip_cases = [d for d in all_dets if d['clip']]

    # conf 불안정 (0.15~0.25)
    unstable_conf = [d for d in all_dets if 0.15 <= d['conf'] < 0.25]

    # 소형 탐지 (E형)
    small_dets = [d for d in all_dets if d['size'] == 'SMALL']

    # 미탐 프레임 분석
    miss_frames = [f for f in frames if f['n_monster'] == 0]

    # 직전 프레임에 탐지 있었는데 현재 없는 케이스 → 갑자기 소실
    sudden_miss = 0
    for i in range(1, len(frames)):
        if frames[i - 1]['n_monster'] > 0 and frames[i]['n_monster'] == 0:
            sudden_miss += 1

    causes = {
        'B_roi_clipping': {
            'count': len(clip_cases),
            'detail': f"탐지된 bbox 중 ROI 경계 근접: {len(clip_cases)}건"
        },
        'E_small_monster': {
            'count': len(small_dets),
            'detail': f"소형 탐지(w*h < {SIZE_SMALL}²): {len(small_dets)}건"
        },
        'H_conf_unstable': {
            'count': len(unstable_conf),
            'detail': f"conf 0.15~0.25 불안정 구간: {len(unstable_conf)}건"
        },
        'sudden_miss': {
            'count': sudden_miss,
            'detail': f"직전 탐지 있었는데 갑자기 소실: {sudden_miss}건"
        },
        'total_miss_frames': len(miss_frames),
        'note': '배경 동화(D)/자세변화(F)/부분가림(G)은 영상 수동 확인 필요'
    }

    # 연속 미탐 트랙 (불안정 트랙)
    high_gap_tracks = [t for t in tracks if t.get('max_gap', 0) >= 5]
    causes['high_gap_tracks'] = len(high_gap_tracks)

    return causes


def sec09_data_gap(frames: List) -> Dict:
    """⑨ 현재 학습 데이터에서 부족한 부분"""
    all_dets = [d for f in frames for d in f['dets']]
    if not all_dets:
        return {}

    total = len(all_dets)

    # 실전 분포 계산
    real_cy = {
        'cy_0_200':   sum(1 for d in all_dets if d['cy'] < 200) / total,
        'cy_200_300': sum(1 for d in all_dets if 200 <= d['cy'] < 300) / total,
        'cy_300_380': sum(1 for d in all_dets if 300 <= d['cy'] < 380) / total,
        'cy_380_472': sum(1 for d in all_dets if d['cy'] >= 380) / total,
    }
    real_cx = {
        'cx_0_100':    sum(1 for d in all_dets if d['cx'] < 100) / total,
        'cx_1000_1135': sum(1 for d in all_dets if d['cx'] >= 1000) / total,
    }

    gaps = {}
    # cy 하단 gap
    train_cy380 = TRAIN_STATS['cy_380_472_pct'] / 100
    real_cy380  = real_cy['cy_380_472']
    gaps['cy_380_472_gap'] = real_cy380 - train_cy380

    # cx 경계 gap
    train_cx0   = TRAIN_STATS['cx_0_100_pct'] / 100
    real_cx0    = real_cx['cx_0_100']
    gaps['cx_0_100_gap'] = real_cx0 - train_cx0

    train_cx1000 = TRAIN_STATS['cx_1000_1135_pct'] / 100
    real_cx1000  = real_cx['cx_1000_1135']
    gaps['cx_1000_1135_gap'] = real_cx1000 - train_cx1000

    return {
        'real_cy_dist':  real_cy,
        'real_cx_edge':  real_cx,
        'gaps':          gaps,
        'train_ref':     TRAIN_STATS,
        'most_critical': '하단(cy≥380) 완전 공백 + 좌우 경계(cx<100, cx>1000) 완전 공백',
    }


def sec10_recommendation(sec01, sec02, sec03, sec04, sec08, sec09) -> List[str]:
    """⑩ 가장 먼저 해야 할 작업 (우선순위 순)"""
    recs = []

    det_rate = sec01.get('det_rate_20', 0)
    avg_conf = sec03.get('avg', 0)
    max_gap  = sec02.get('max_gap_frames', 0)
    unstable = sec03.get('unstable_15_25', 0)
    cy380_gap = sec09.get('gaps', {}).get('cy_380_472_gap', 0)
    cx_gap    = max(
        sec09.get('gaps', {}).get('cx_0_100_gap', 0),
        sec09.get('gaps', {}).get('cx_1000_1135_gap', 0)
    )
    clip_cnt  = sec08.get('B_roi_clipping', {}).get('count', 0)
    small_cnt = sec08.get('E_small_monster', {}).get('count', 0)

    # 우선순위 결정
    if cy380_gap > 0.15:
        recs.append(
            "🔴 [1순위] 하단부(cy≥380) 학습 데이터 추가\n"
            f"   실전 {cy380_gap*100:.0f}% 발생 vs 학습 데이터 0% → 200장 이상 캡처 후 재학습\n"
            "   명령어: (게임 필드에서 몬스터 하단부 위치 시 스크린샷 + labelImg annotation)"
        )
    if cx_gap > 0.10:
        recs.append(
            "🔴 [2순위] 좌우 경계(cx<100, cx>1000) 학습 데이터 추가\n"
            f"   실전 경계 발생 {cx_gap*100:.0f}% vs 학습 데이터 0% → 각 80장 캡처 후 재학습"
        )
    if det_rate < 0.70:
        recs.append(
            f"🟠 [3순위] 탐지율 {det_rate*100:.0f}% 개선 필요\n"
            "   단기: config.json min_conf: 0.20 → 0.15 (즉시 적용, 재학습 불필요)\n"
            "   장기: 데이터 추가 후 재학습"
        )
    if max_gap >= 8:
        recs.append(
            f"🟠 [4순위] 연속 미탐 최대 {max_gap}프레임 — 깜빡임 해소\n"
            "   원인: 이동 자세 변화 시 conf 급락\n"
            "   대책: 이동/공격 자세 학습 데이터 50장 추가"
        )
    if clip_cnt > 10:
        recs.append(
            f"🟡 [5순위] ROI 경계 clipping {clip_cnt}건\n"
            "   대책: ROI width/height 10~20px 확장 (config.json roi 조정)\n"
            "   또는: 경계 걸린 몬스터 annotation 추가"
        )
    if unstable > 0.20:
        recs.append(
            f"🟡 [6순위] conf 0.15~0.25 불안정 구간 {unstable*100:.0f}%\n"
            "   원인: 학습 데이터 해당 위치/배경 부족\n"
            "   대책: 1~2순위 데이터 추가 후 자연 해소 기대"
        )
    if not recs:
        recs.append("✅ 현재 성능 양호 — 실전 테스트 결과 기반 미세 조정 권장")

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# 보고서 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_report(sess_dir: str, frames: List, tracks: List):
    sep = "─" * 58

    print()
    print("=" * 58)
    print("  YOLO 탐지 검증 보고서")
    print(f"  세션: {os.path.basename(sess_dir)}")
    print("=" * 58)
    print()

    # 섹션별 계산
    s01 = sec01_overall(frames, tracks)
    s02 = sec02_stability(tracks)
    s03 = sec03_confidence(frames)
    s04 = sec04_zone(frames)
    s05 = sec05_size(frames)
    s06 = sec06_background(frames)
    s07 = sec07_movement(frames, tracks)
    s08 = sec08_miss_causes(frames, tracks)
    s09 = sec09_data_gap(frames)
    s10 = sec10_recommendation(s01, s02, s03, s04, s08, s09)

    # ── ① 탐지율 ────────────────────────────────────────────────────
    print(f"① 실제 몬스터 탐지율")
    print(sep)
    print(f"  총 프레임:        {s01['total_frames']}")
    print(f"  탐지 (conf≥0.20): {s01['det_frames_20']} "
          f"({s01['det_rate_20']*100:.1f}%)")
    print(f"  탐지 (conf≥0.15): {s01['det_frames_15']} "
          f"({s01['det_rate_15']*100:.1f}%)")
    print(f"  탐지 (아무거나):  {s01['det_frames_any']} "
          f"({s01['det_rate_any']*100:.1f}%)")
    print(f"  미탐 프레임:      {s01['miss_frames']}")
    print(f"  총 탐지 건수:     {s01['total_dets']}")
    print()

    # ── ② 연속 탐지 안정성 ──────────────────────────────────────────
    print(f"② 연속 탐지 안정성")
    print(sep)
    if s02.get('total_tracks', 0) == 0:
        print("  트랙 없음 — 탐지 불충분")
    else:
        print(f"  총 트랙 수:       {s02['total_tracks']}")
        print(f"  평균 recall:      {s02['avg_recall']*100:.1f}%")
        print(f"  최소 recall:      {s02['min_recall']*100:.1f}%")
        print(f"  최대 연속 미탐:   {s02['max_gap_frames']}프레임")
        print(f"  평균 미탐 간격:   {s02['avg_gap']:.1f}프레임")
        print(f"  안정 트랙(≥80%): {s02['stable_tracks']}")
        print(f"  불안정(<50%):    {s02['unstable_tracks']}")
    print()

    # ── ③ confidence 분포 ────────────────────────────────────────────
    print(f"③ confidence 분포")
    print(sep)
    if s03.get('count', 0) == 0:
        print("  탐지 없음")
    else:
        print(f"  평균: {s03['avg']:.3f}  std: {s03['std']:.3f}  "
              f"min: {s03['min']:.3f}  max: {s03['max']:.3f}")
        print(f"  conf≥0.20:  {s03['pct_gte_20']*100:.1f}%")
        print(f"  conf≥0.15:  {s03['pct_gte_15']*100:.1f}%")
        print(f"  불안정구간(0.15~0.25): {s03['unstable_15_25']*100:.1f}%")
        print()
        print("  히스토그램:")
        total_c = s03['count']
        for rng, cnt in s03['histogram'].items():
            bar = '█' * min(30, cnt // max(1, total_c // 30))
            print(f"    [{rng}]: {cnt:4d} ({cnt/total_c*100:4.1f}%) {bar}")
    print()

    # ── ④ 위치별 탐지율 ─────────────────────────────────────────────
    print(f"④ 위치별 탐지율 (ROI 기준)")
    print(sep)
    if not s04:
        print("  탐지 없음")
    else:
        all_dets_total = sum(v['count'] for k, v in s04.items()
                             if isinstance(v, dict) and 'count' in v)
        for zone, info in sorted(s04.items(), key=lambda x: -x[1].get('count', 0)
                                 if isinstance(x[1], dict) else 0):
            if not isinstance(info, dict) or 'count' not in info:
                continue
            print(f"  {zone:<25} {info['count']:4d} "
                  f"({info['pct']*100:4.1f}%)  "
                  f"avg_conf={info['avg_conf']:.3f}")
        print(f"  경계 탐지 비율: {s04.get('_edge_pct',0)*100:.1f}%")
        print(f"  하단 탐지 비율: {s04.get('_btm_pct',0)*100:.1f}%")
        print(f"  경계 clipping:  {s04.get('_clip_count',0)}건")
    print()

    # ── ⑤ 크기별 탐지율 ─────────────────────────────────────────────
    print(f"⑤ 몬스터 크기별 탐지율")
    print(sep)
    if not s05 or 'avg_w' not in s05:
        print("  탐지 없음")
    else:
        print(f"  실전 bbox: avg_w={s05['avg_w']:.1f}px "
              f"avg_h={s05['avg_h']:.1f}px")
        print(f"             min_w={s05['min_w']:.1f}px "
              f"min_h={s05['min_h']:.1f}px")
        print(f"  학습데이터: avg_w={TRAIN_STATS['avg_w']}px "
              f"avg_h={TRAIN_STATS['avg_h']}px")
        for sz in ['SMALL', 'MID', 'LARGE']:
            if sz in s05:
                v = s05[sz]
                print(f"  {sz:6}: {v['count']:4d} ({v['pct']*100:4.1f}%)  "
                      f"avg_conf={v['avg_conf']:.3f}  "
                      f"avg_area={v['avg_area']:.0f}px²")
    print()

    # ── ⑥ 배경별 탐지율 ─────────────────────────────────────────────
    print(f"⑥ 배경별 탐지율 (cy 기반 추론)")
    print(sep)
    if 'note' in s06:
        print(f"  ※ {s06['note']}")
    for label, v in s06.items():
        if label == 'note' or not isinstance(v, dict):
            continue
        print(f"  {label}: {v['count']}건 ({v['pct']*100:.1f}%)  "
              f"avg_conf={v['avg_conf']:.3f}")
    print()

    # ── ⑦ 이동 중 탐지율 ────────────────────────────────────────────
    print(f"⑦ 이동 중 탐지율")
    print(sep)
    print(f"  {s07.get('note','')}")
    print(f"  이동 중 탐지:    {s07.get('moving_det',0)}건")
    print(f"  이동 중 미탐:    {s07.get('moving_miss',0)}건")
    print(f"  정지 중 탐지:    {s07.get('stationary_det',0)}건")
    dr_move = s07.get('det_rate_moving', 0)
    dr_stat = s07.get('det_rate_stationary', 0)
    print(f"  이동 중 탐지율:  {dr_move*100:.1f}%")
    print(f"  정지 중 탐지율:  {dr_stat*100:.1f}%")
    if dr_move > 0 and dr_stat > 0:
        diff = (dr_stat - dr_move) * 100
        print(f"  이동/정지 차이:  {diff:+.1f}%p  "
              f"{'(이동 중 성능 저하 있음)' if diff > 5 else '(이동 영향 없음)'}")
    print()

    # ── ⑧ 미탐 원인 분류 ────────────────────────────────────────────
    print(f"⑧ 미탐 원인 분류")
    print(sep)
    causes_order = [
        ('B_roi_clipping', 'B. ROI 경계 clipping'),
        ('E_small_monster', 'E. 너무 작은 몬스터'),
        ('H_conf_unstable', 'H. conf 불안정(0.15~0.25)'),
        ('sudden_miss',     '갑작스런 소실'),
    ]
    for key, label in causes_order:
        v = s08.get(key, {})
        if isinstance(v, dict):
            print(f"  {label}: {v.get('count',0)}건")
            if 'detail' in v:
                print(f"    → {v['detail']}")
        else:
            print(f"  {label}: {v}건")
    print(f"  총 미탐 프레임:  {s08.get('total_miss_frames',0)}")
    print(f"  불안정 트랙:     {s08.get('high_gap_tracks',0)}")
    print(f"  ※ D/F/G형 (배경동화/자세변화/부분가림) → 영상 수동 확인 필요")
    print()

    # ── ⑨ 학습 데이터 부족 ──────────────────────────────────────────
    print(f"⑨ 현재 학습 데이터에서 부족한 부분")
    print(sep)
    if not s09:
        print("  분석 불가 — 탐지 데이터 부족")
    else:
        print(f"  실전 cy 분포:")
        for k, v in s09.get('real_cy_dist', {}).items():
            train_key = k + '_pct'
            train_val = TRAIN_STATS.get(train_key, 0)
            gap = v - train_val / 100
            flag = " ⚠️ 심각" if gap > 0.15 else (" ⚠️" if gap > 0.05 else "")
            print(f"    {k}: 실전={v*100:.1f}%  학습={train_val:.1f}%  "
                  f"gap={gap*100:+.1f}%{flag}")
        print(f"  실전 cx 경계:")
        for k, v in s09.get('real_cx_edge', {}).items():
            train_key = k + '_pct'
            train_val = TRAIN_STATS.get(train_key, 0)
            gap = v - train_val / 100
            flag = " ⚠️ 심각" if gap > 0.10 else ""
            print(f"    {k}: 실전={v*100:.1f}%  학습={train_val:.1f}%  "
                  f"gap={gap*100:+.1f}%{flag}")
        print(f"  가장 심각: {s09.get('most_critical','')}")
    print()

    # ── ⑩ 권장 작업 ─────────────────────────────────────────────────
    print(f"⑩ 탐지 성능을 높이기 위해 가장 먼저 해야 할 작업")
    print(sep)
    for i, rec in enumerate(s10, 1):
        print(f"  {rec}")
        print()
    print()

    # ── 저장 ─────────────────────────────────────────────────────────
    report = {
        'session': os.path.basename(sess_dir),
        'generated': datetime.now().isoformat(),
        'sec01_overall': s01,
        'sec02_stability': s02,
        'sec03_confidence': s03,
        'sec04_zone': {k: v for k, v in s04.items()},
        'sec05_size': s05,
        'sec06_background': s06,
        'sec07_movement': s07,
        'sec08_miss_causes': s08,
        'sec09_data_gap': s09,
        'sec10_recommendations': s10,
    }
    report_path = os.path.join(sess_dir, 'detection_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  보고서 저장: {report_path}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 엔트리포인트
# ══════════════════════════════════════════════════════════════════════════════

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'detection_sessions')

    if len(sys.argv) >= 2:
        sess_dir = sys.argv[1]
        if not os.path.isabs(sess_dir):
            sess_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    sess_dir)
    else:
        sess_dir = latest_session(base_dir)
        print(f"[AUTO] 최신 세션: {sess_dir}")

    frames, tracks = load_session(sess_dir)
    print(f"[LOAD] frames={len(frames)}  tracks={len(tracks)}")

    print_report(sess_dir, frames, tracks)


if __name__ == '__main__':
    main()
