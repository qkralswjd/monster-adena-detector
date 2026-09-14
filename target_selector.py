"""
target_selector.py
------------------
현재 탐지된 몬스터 목록에서 타겟을 선택하는 로직.

선택 방식:
  - nearest  : 화면 중앙에서 가장 가까운 몬스터
  - click    : 클릭 위치에서 가장 가까운 몬스터
"""

import math
from typing import List, Optional, Tuple
from detector import Detection


def _dist(ax: int, ay: int, bx: int, by: int) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


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


def find_matching(detections: List[Detection],
                  target: Detection,
                  iou_thresh: float = 0.3) -> Optional[Detection]:
    """현재 탐지 목록에서 이전 타겟과 IoU가 가장 높은 것 반환.
    thresh 미만이면 None (타겟 소실/사망 판정)."""
    if not detections or target is None:
        return None
    best = max(detections, key=lambda d: iou(d, target))
    if iou(best, target) >= iou_thresh:
        return best
    return None
