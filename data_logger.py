"""
data_logger.py
--------------
타겟 추적 데이터를 기록하는 모듈.

기록 항목:
  - timestamp, 좌표, bbox, 속도, 방향
  - detection/tracking/target confidence
  - 상태 (tracking/lost/confirmed_lost)
"""

import csv
import os
import time
import math
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class LogRecord:
    timestamp: float
    frame_id: int
    track_id: int
    cx: int
    cy: int
    w: int
    h: int
    vx: float
    vy: float
    speed: float
    direction_deg: float      # 이동 방향 (0=오른쪽, 90=아래)
    det_confidence: float
    track_confidence: float
    target_confidence: float
    state: str


class DataLogger:
    def __init__(self, save_dir: str = "logs", max_records: int = 10000):
        self._save_dir = save_dir
        self._max_records = max_records
        self._records: List[LogRecord] = []
        self._frame_id = 0
        os.makedirs(save_dir, exist_ok=True)

    def log(self, target_info):
        """TargetInfo를 받아서 기록."""
        if target_info is None:
            self._frame_id += 1
            return

        direction = math.degrees(math.atan2(target_info.vy, target_info.vx)) \
                    if (target_info.vx != 0 or target_info.vy != 0) else 0.0

        record = LogRecord(
            timestamp        = time.time(),
            frame_id         = self._frame_id,
            track_id         = target_info.track_id,
            cx               = target_info.cx,
            cy               = target_info.cy,
            w                = target_info.w,
            h                = target_info.h,
            vx               = round(target_info.vx, 2),
            vy               = round(target_info.vy, 2),
            speed            = round(target_info.speed, 2),
            direction_deg    = round(direction, 1),
            det_confidence   = round(target_info.det_confidence, 3),
            track_confidence = round(target_info.track_confidence, 3),
            target_confidence= round(target_info.target_confidence, 3),
            state            = target_info.state,
        )
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records.pop(0)
        self._frame_id += 1

    def save_csv(self, filename: Optional[str] = None):
        """CSV로 저장."""
        if not self._records:
            return
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"track_{ts}.csv"
        path = os.path.join(self._save_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self._records[0]).keys())
            writer.writeheader()
            for r in self._records:
                writer.writerow(asdict(r))
        print(f"[Logger] {len(self._records)}개 기록 저장: {path}")

    def get_path_points(self, last_n: int = 100):
        """최근 N개의 (cx, cy) 리스트 반환 (경로 표시용)."""
        recs = self._records[-last_n:]
        return [(r.cx, r.cy) for r in recs]

    def clear(self):
        self._records.clear()
        self._frame_id = 0

    @property
    def record_count(self) -> int:
        return len(self._records)
