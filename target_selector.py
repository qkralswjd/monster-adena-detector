"""
target_selector.py
------------------
탐지된 Detection 목록에서 타겟 하나를 선택.

현재 전략:
  - confidence 가 가장 높은 몬스터 선택
  - 추후 nearest / manual 등 전략 추가 가능

좌표 원칙:
  - 입력/출력 모두 화면 절대좌표 그대로
  - 변환 없음
"""

import math
from typing import List, Optional
from detector import Detection


def select_by_confidence(detections: List[Detection]) -> Optional[Detection]:
    """
    confidence 가 가장 높은 Detection 반환.
    없으면 None.
    """
    monsters = [d for d in detections if d.class_id == 0]
    if not monsters:
        return None
    return max(monsters, key=lambda d: d.confidence)


def select_nearest(detections: List[Detection],
                   ref_x: int, ref_y: int) -> Optional[Detection]:
    """
    기준점 (ref_x, ref_y) 에서 가장 가까운 몬스터 반환.
    없으면 None.
    """
    monsters = [d for d in detections if d.class_id == 0]
    if not monsters:
        return None
    return min(monsters,
               key=lambda d: math.hypot(d.cx - ref_x, d.cy - ref_y))
