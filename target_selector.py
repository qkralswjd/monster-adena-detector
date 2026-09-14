"""
target_selector.py
------------------
탐지된 몬스터 목록에서 타겟을 선택하고 추적하는 로직.

선택 방식:
  - nearest  : 화면 중앙에서 가장 가까운 몬스터
  - click    : 클릭 위치에서 가장 가까운 몬스터

추적 방식 (find_matching):
  - IoU + 거리 혼합 스코어로 프레임 간 동일 몬스터 매칭
  - IoU 0 이어도 거리가 가까우면 추적 유지 (이동 중 박스 겹침 없을 때)
  - 박스 크기 변화율로 완전히 다른 몬스터 배제
"""

import math
from typing import List, Optional
from detector import Detection


# ══════════════════════════════════════════════════════════════
#  기본 유틸
# ══════════════════════════════════════════════════════════════

def _dist(ax: int, ay: int, bx: int, by: int) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ══════════════════════════════════════════════════════════════
#  IoU
# ══════════════════════════════════════════════════════════════

def iou(a: Detection, b: Detection) -> float:
    """두 Detection의 IoU 계산."""
    ax1, ay1, ax2, ay2 = a.tlbr
    bx1, by1, bx2, by2 = b.tlbr

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter == 0:
        return 0.0

    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  선택
# ══════════════════════════════════════════════════════════════

def select_nearest(detections: List[Detection],
                   cx: int, cy: int) -> Optional[Detection]:
    """기준점(cx, cy)에서 가장 가까운 몬스터 반환."""
    if not detections:
        return None
    return min(detections, key=lambda d: _dist(d.cx, d.cy, cx, cy))


def select_by_click(detections: List[Detection],
                    click_x: int, click_y: int,
                    radius: int = 60) -> Optional[Detection]:
    """클릭 위치 반경 안에서 가장 가까운 몬스터 반환.
    반경 안에 없으면 None."""
    candidates = [
        d for d in detections
        if _dist(d.cx, d.cy, click_x, click_y) <= radius
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda d: _dist(d.cx, d.cy, click_x, click_y))


# ══════════════════════════════════════════════════════════════
#  추적: IoU + 거리 혼합 스코어
# ══════════════════════════════════════════════════════════════

def _match_score(candidate: Detection,
                 target: Detection,
                 max_dist: float,
                 iou_weight: float = 0.5,
                 dist_weight: float = 0.5) -> float:
    """
    매칭 스코어 계산 (높을수록 같은 몬스터).

    score = iou_weight * iou_score
          + dist_weight * (1 - dist / max_dist 클리핑)

    iou_score  : 두 박스의 IoU (0~1)
    dist_score : 중심 거리 기반 유사도 (0~1, 가까울수록 1)
    max_dist   : 탐색 반경 (px). 이 이상 떨어지면 dist_score=0
    """
    iou_score  = iou(candidate, target)
    dist_val   = _dist(candidate.cx, candidate.cy, target.cx, target.cy)
    dist_score = max(0.0, 1.0 - dist_val / max(max_dist, 1.0))
    return iou_weight * iou_score + dist_weight * dist_score


def find_matching(detections: List[Detection],
                  target: Detection,
                  iou_thresh: float = 0.3,
                  max_dist: float = 120.0,
                  min_score: float = 0.25,
                  iou_weight: float = 0.5,
                  dist_weight: float = 0.5) -> Optional[Detection]:
    """
    현재 탐지 목록에서 이전 타겟과 가장 유사한 것을 반환.

    판정 우선순위:
      1. IoU ≥ iou_thresh  →  IoU 기준 최고 매칭 바로 채택
         (제자리 또는 소폭 이동 시 기존 로직과 동일)
      2. IoU < iou_thresh  →  혼합 스코어가 min_score 이상인 최고 후보 채택
         (빠르게 이동해서 박스가 안 겹칠 때도 거리로 추적)
      3. 둘 다 실패  →  None (소실/사망)

    파라미터:
      iou_thresh  : IoU 우선 채택 최소값 (기본 0.3)
      max_dist    : 거리 스코어 계산 최대 반경 (기본 120px)
      min_score   : 혼합 스코어 최소 임계값 (기본 0.25)
      iou_weight  : IoU 가중치 (기본 0.5)
      dist_weight : 거리 가중치 (기본 0.5)
    """
    if not detections or target is None:
        return None

    # 1) IoU 우선: iou_thresh 넘는 후보 있으면 바로 사용
    best_iou = max(detections, key=lambda d: iou(d, target))
    best_iou_val = iou(best_iou, target)
    if best_iou_val >= iou_thresh:
        return best_iou

    # 2) 혼합 스코어: 이동 중 박스 겹침 없는 경우
    best_mix = max(
        detections,
        key=lambda d: _match_score(d, target, max_dist, iou_weight, dist_weight)
    )
    best_mix_score = _match_score(best_mix, target, max_dist, iou_weight, dist_weight)

    if best_mix_score >= min_score:
        return best_mix

    # 3) 완전 소실
    return None
