"""
byte_tracker.py
---------------
ByteTrack 기반 다중 몬스터 ID 추적 모듈.

핵심 아이디어:
  - High confidence 탐지 → 우선 매칭
  - Low confidence 탐지 → 사라진 트랙에 2차 매칭 (놓친 몬스터 복구)
  - 칼만 필터로 위치 예측 → 빠르게 움직여도 ID 유지
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from detector import Detection
from kalman_filter import KalmanFilter


# ------------------------------------------------------------------
# Track 상태
# ------------------------------------------------------------------
TRACK_NEW      = "new"
TRACK_TRACKED  = "tracked"
TRACK_LOST     = "lost"
TRACK_REMOVED  = "removed"


@dataclass
class Track:
    """추적 중인 몬스터 1개."""
    track_id: int
    det: Detection

    state: str = TRACK_NEW
    age: int = 0                    # 등록 후 총 프레임 수
    hit_streak: int = 0             # 연속 탐지된 프레임 수
    lost_frames: int = 0            # 연속 미탐지 프레임 수
    score: float = 0.0

    # 칼만 필터 상태
    mean: Optional[np.ndarray] = field(default=None, repr=False)
    covariance: Optional[np.ndarray] = field(default=None, repr=False)

    # 이동 기록
    history: List[Tuple[int,int]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.score = self.det.confidence
        self.history.append((self.det.cx, self.det.cy))

    @property
    def cx(self) -> int:
        if self.mean is not None:
            return int(self.mean[0])
        return self.det.cx

    @property
    def cy(self) -> int:
        if self.mean is not None:
            return int(self.mean[1])
        return self.det.cy

    @property
    def predicted_bbox(self) -> Tuple[int,int,int,int]:
        """칼만 예측 bbox (x,y,w,h)."""
        if self.mean is not None:
            cx, cy = int(self.mean[0]), int(self.mean[1])
            w, h = int(self.mean[2]), int(self.mean[3])
            return (cx - w//2, cy - h//2, w, h)
        return self.det.bbox

    @property
    def velocity(self) -> Tuple[float, float]:
        """현재 이동 속도 (vx, vy)."""
        if self.mean is not None and len(self.mean) >= 6:
            return (float(self.mean[4]), float(self.mean[5]))
        return (0.0, 0.0)

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.sqrt(vx**2 + vy**2))


def _iou(a: Detection, b_mean: np.ndarray) -> float:
    """Detection과 칼만 예측 박스의 IoU."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx = int(b_mean[0] - b_mean[2]/2)
    by = int(b_mean[1] - b_mean[3]/2)
    bx2 = bx + int(b_mean[2])
    by2 = by + int(b_mean[3])
    ix1, iy1 = max(ax1, bx), max(ay1, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1) * (iy2-iy1)
    union = a.w*a.h + int(b_mean[2])*int(b_mean[3]) - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(cost_matrix: np.ndarray,
                  thresh: float) -> Tuple[List,List,List]:
    """헝가리안 매칭 (lap 없으면 greedy fallback)."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    try:
        import lap
        _, row_idx, col_idx = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [(r, col_idx[r]) for r in range(len(row_idx))
                   if col_idx[r] >= 0 and cost_matrix[r, col_idx[r]] <= thresh]
        unmatched_a = [r for r in range(cost_matrix.shape[0])
                       if not any(m[0]==r for m in matches)]
        unmatched_b = [c for c in range(cost_matrix.shape[1])
                       if not any(m[1]==c for m in matches)]
        return matches, unmatched_a, unmatched_b
    except ImportError:
        # greedy fallback
        used_r, used_c = set(), set()
        pairs = sorted([(cost_matrix[r,c], r, c)
                        for r in range(cost_matrix.shape[0])
                        for c in range(cost_matrix.shape[1])
                        if cost_matrix[r,c] <= thresh])
        matches = []
        for _, r, c in pairs:
            if r not in used_r and c not in used_c:
                matches.append((r, c))
                used_r.add(r); used_c.add(c)
        unmatched_a = [r for r in range(cost_matrix.shape[0]) if r not in used_r]
        unmatched_b = [c for c in range(cost_matrix.shape[1]) if c not in used_c]
        return matches, unmatched_a, unmatched_b


class ByteTracker:
    """
    ByteTrack 알고리즘으로 다중 몬스터 ID 추적.

    흐름:
      1. 칼만 필터로 기존 트랙 위치 예측
      2. High conf 탐지 → 기존 트랙과 IoU 매칭
      3. Low conf 탐지 → 미매칭 트랙과 2차 매칭
      4. 매칭 안된 탐지 → 새 트랙 생성
      5. 오래 사라진 트랙 → 제거
    """

    def __init__(self,
                 track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 max_lost_frames: int = 60):
        self._high_thresh = track_high_thresh
        self._low_thresh = track_low_thresh
        self._new_thresh = new_track_thresh
        self._buffer = track_buffer
        self._match_thresh = match_thresh
        self._max_lost = max_lost_frames

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._kf = KalmanFilter()
        self._frame_count = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        새 탐지 결과로 트랙 업데이트.
        Returns: 현재 활성 트랙 리스트
        """
        self._frame_count += 1

        # ── 1. 칼만 예측 ──────────────────────────────────────────────
        for t in self._tracks.values():
            if t.mean is not None:
                t.mean, t.covariance = self._kf.predict(t.mean, t.covariance)
            t.age += 1

        # ── 2. High/Low conf 분리 ─────────────────────────────────────
        high_dets = [d for d in detections if d.confidence >= self._high_thresh]
        low_dets  = [d for d in detections if self._low_thresh <= d.confidence < self._high_thresh]

        active = [t for t in self._tracks.values() if t.state == TRACK_TRACKED]
        lost   = [t for t in self._tracks.values() if t.state == TRACK_LOST]

        # ── 3. 1차 매칭: active 트랙 ↔ high conf 탐지 ────────────────
        matches1, unmatched_tracks1, unmatched_dets1 = \
            self._match_tracks(active, high_dets, self._match_thresh)

        for ti, di in matches1:
            self._update_track(active[ti], high_dets[di])

        # ── 4. 2차 매칭: 미매칭 트랙 ↔ low conf 탐지 ────────────────
        remaining_tracks = [active[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2, unmatched_dets2 = \
            self._match_tracks(remaining_tracks, low_dets, 0.5)

        for ti, di in matches2:
            self._update_track(remaining_tracks[ti], low_dets[di])

        # ── 5. lost 트랙 ↔ 미매칭 high conf 탐지 재매칭 ─────────────
        remaining_high = [high_dets[i] for i in unmatched_dets1]
        matches3, _, unmatched_new = \
            self._match_tracks(lost, remaining_high, 0.7)

        for ti, di in matches3:
            self._recover_track(lost[ti], remaining_high[di])

        # ── 6. 새 트랙 생성 ───────────────────────────────────────────
        new_dets = [remaining_high[i] for i in unmatched_new]
        for det in new_dets:
            if det.confidence >= self._new_thresh:
                self._create_track(det)

        # ── 7. 미매칭 트랙 → lost 처리 ───────────────────────────────
        still_unmatched = [remaining_tracks[i] for i in unmatched_tracks2]
        for t in still_unmatched:
            t.state = TRACK_LOST
            t.lost_frames += 1
            t.hit_streak = 0

        for t in lost:
            if not any(m[0] == lost.index(t) for m in matches3):
                t.lost_frames += 1

        # ── 8. 오래된 lost 트랙 제거 ──────────────────────────────────
        to_remove = [tid for tid, t in self._tracks.items()
                     if t.lost_frames > self._max_lost]
        for tid in to_remove:
            self._tracks[tid].state = TRACK_REMOVED
            del self._tracks[tid]

        return [t for t in self._tracks.values()
                if t.state in (TRACK_TRACKED, TRACK_NEW)]

    def _match_tracks(self, tracks: List[Track],
                      dets: List[Detection],
                      thresh: float):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        cost = np.zeros((len(tracks), len(dets)))
        for ti, t in enumerate(tracks):
            for di, d in enumerate(dets):
                if t.mean is not None:
                    iou = _iou(d, t.mean)
                else:
                    # 칼만 없으면 거리 기반
                    dist = np.sqrt((t.cx-d.cx)**2 + (t.cy-d.cy)**2)
                    iou = max(0, 1 - dist/300)
                cost[ti, di] = 1 - iou

        return _greedy_match(cost, 1 - thresh)

    def _update_track(self, track: Track, det: Detection):
        meas = np.array([det.cx, det.cy, det.w, det.h], dtype=float)
        if track.mean is None:
            track.mean, track.covariance = self._kf.initiate(meas)
        else:
            track.mean, track.covariance = self._kf.update(
                track.mean, track.covariance, meas)
        track.det = det
        track.score = det.confidence
        track.state = TRACK_TRACKED
        track.lost_frames = 0
        track.hit_streak += 1
        track.history.append((det.cx, det.cy))
        if len(track.history) > 100:
            track.history.pop(0)

    def _recover_track(self, track: Track, det: Detection):
        self._update_track(track, det)
        track.state = TRACK_TRACKED
        print(f"[ByteTracker] Track #{track.track_id} 복구")

    def _create_track(self, det: Detection):
        tid = self._next_id
        self._next_id += 1
        t = Track(track_id=tid, det=det, state=TRACK_NEW, score=det.confidence)
        meas = np.array([det.cx, det.cy, det.w, det.h], dtype=float)
        t.mean, t.covariance = self._kf.initiate(meas)
        t.hit_streak = 1
        self._tracks[tid] = t

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def clear(self):
        self._tracks.clear()
        self._next_id = 1

    @property
    def track_count(self) -> int:
        return len(self._tracks)
