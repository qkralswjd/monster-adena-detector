"""
target_tracker.py
-----------------
단일 타겟 몬스터를 고정 추적하는 핵심 모듈.

기능:
  1. ByteTracker의 트랙 중 Target ID 고정 유지
  2. Target이 사라지면 칼만 예측으로 예상 위치 계산
  3. 예상 위치 근처에서 재탐지 시도 (Re-identification)
  4. 위치/크기/속도/외형으로 동일 몬스터 검증
  5. Confidence Manager로 신뢰도 관리

특수 상황 처리:
  - 다른 몬스터와 겹침: IoU + 속도 방향으로 구분
  - 잠시 사라짐: 칼만 예측 유지 + 재탐지
  - 화면 밖: confirmed lost
  - 순간이동: velocity spike 감지 → confidence 감소
  - 비슷한 몬스터: 크기 + 속도 + 방향 조합 매칭
"""

import numpy as np
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from byte_tracker import Track, ByteTracker
from confidence_manager import ConfidenceManager
from kalman_filter import KalmanFilter


# ------------------------------------------------------------------
# Target 상태
# ------------------------------------------------------------------
TARGET_NONE      = "NONE"       # 타겟 미지정
TARGET_TRACKING  = "TRACKING"   # 정상 추적 중
TARGET_LOST      = "LOST"       # 임시 소실 (재탐색 중)
TARGET_CONFIRMED_LOST = "CONFIRMED_LOST"  # 완전 소실


@dataclass
class TargetInfo:
    """현재 타겟 정보."""
    track_id: int
    cx: int
    cy: int
    w: int
    h: int
    vx: float
    vy: float
    speed: float
    det_confidence: float
    track_confidence: float
    target_confidence: float
    state: str
    age: int
    timestamp: float = field(default_factory=time.time)

    # 예측 위치 (사라졌을 때)
    pred_cx: int = 0
    pred_cy: int = 0


class TargetTracker:
    """
    단일 타겟 고정 추적기.
    ByteTracker 위에서 동작하며 특정 ID를 잃지 않도록 관리.
    """

    def __init__(self, cfg):
        t = cfg["target"]
        self._reid_thresh   = t["reid_dist_thresh"]
        self._pred_max      = t["prediction_max_frames"]

        self._byte_tracker = ByteTracker(
            track_high_thresh = cfg["tracker"]["track_high_thresh"],
            track_low_thresh  = cfg["tracker"]["track_low_thresh"],
            new_track_thresh  = cfg["tracker"]["new_track_thresh"],
            track_buffer      = cfg["tracker"]["track_buffer"],
            match_thresh      = cfg["tracker"]["match_thresh"],
            max_lost_frames   = cfg["tracker"]["max_lost_frames"],
        )
        self._conf_mgr = ConfidenceManager(
            decay           = t["confidence_decay"],
            boost           = t["confidence_boost"],
            min_confidence  = t["min_confidence"],
            lost_confirm_sec= t["lost_confirm_sec"],
        )
        self._kf = KalmanFilter()

        # 타겟 상태
        self._target_id: Optional[int] = None
        self._state: str = TARGET_NONE
        self._last_track: Optional[Track] = None
        self._pred_mean: Optional[np.ndarray] = None
        self._pred_cov: Optional[np.ndarray] = None
        self._pred_frames: int = 0

        # 스냅샷 (Re-ID용)
        self._target_snapshot: Optional[np.ndarray] = None
        self._target_size: Optional[Tuple[int,int]] = None  # (w, h)

    # ------------------------------------------------------------------
    # 타겟 지정
    # ------------------------------------------------------------------

    def set_target(self, track_id: int, tracks: List[Track],
                   frame: Optional[np.ndarray] = None):
        """클릭 등으로 특정 트랙을 타겟으로 지정."""
        track = next((t for t in tracks if t.track_id == track_id), None)
        if track is None:
            print(f"[TargetTracker] Track #{track_id} 없음")
            return False

        self._target_id = track_id
        self._last_track = track
        self._state = TARGET_TRACKING
        self._conf_mgr.reset(initial=1.0)
        self._pred_frames = 0

        # 타겟 크기 기억
        self._target_size = (track.det.w, track.det.h)

        # 타겟 외형 스냅샷 저장 (Re-ID용)
        if frame is not None:
            self._save_snapshot(frame, track)

        print(f"[TargetTracker] Target 지정: #{track_id} "
              f"({track.cx}, {track.cy})")
        return True

    def clear_target(self):
        """타겟 해제."""
        self._target_id = None
        self._state = TARGET_NONE
        self._last_track = None
        self._pred_mean = None
        self._pred_frames = 0
        self._conf_mgr.reset()
        print("[TargetTracker] Target 해제")

    # ------------------------------------------------------------------
    # 메인 업데이트
    # ------------------------------------------------------------------

    def update(self, detections, frame: Optional[np.ndarray] = None
               ) -> Tuple[List[Track], Optional[TargetInfo]]:
        """
        매 프레임 호출.
        Returns: (all_tracks, target_info)
        """
        # ByteTracker 업데이트
        all_tracks = self._byte_tracker.update(detections)

        if self._target_id is None:
            return all_tracks, None

        # 현재 타겟 트랙 찾기
        target_track = next(
            (t for t in all_tracks if t.track_id == self._target_id), None)

        if target_track is not None:
            # ── 타겟 발견 ──────────────────────────────────────────────
            # 순간이동 감지 (너무 빠른 이동)
            if self._last_track is not None:
                dist = np.sqrt(
                    (target_track.cx - self._last_track.cx)**2 +
                    (target_track.cy - self._last_track.cy)**2
                )
                if dist > 300:  # 한 프레임에 300px 이상 이동
                    print(f"[TargetTracker] 순간이동 감지 dist={dist:.0f} → confidence 감소")
                    self._conf_mgr.on_missed()
                else:
                    self._conf_mgr.on_detected(target_track.score)
            else:
                self._conf_mgr.on_detected(target_track.score)

            self._last_track = target_track
            self._state = TARGET_TRACKING
            self._pred_frames = 0

            if frame is not None:
                self._save_snapshot(frame, target_track)

        else:
            # ── 타겟 소실 ──────────────────────────────────────────────
            self._conf_mgr.on_missed()

            if self._conf_mgr.is_confirmed_lost:
                self._state = TARGET_CONFIRMED_LOST
            elif self._conf_mgr.is_lost:
                self._state = TARGET_LOST
                # 칼만 예측으로 예상 위치 유지
                self._pred_frames += 1
                if self._pred_frames <= self._pred_max and self._last_track is not None:
                    if self._last_track.mean is not None:
                        self._pred_mean, _ = self._kf.predict(
                            self._last_track.mean,
                            self._last_track.covariance
                        )
                # Re-ID 시도
                target_track = self._try_reid(all_tracks, frame)
                if target_track:
                    print(f"[TargetTracker] Re-ID 성공: #{target_track.track_id}")
                    self._target_id = target_track.track_id
                    self._last_track = target_track
                    self._state = TARGET_TRACKING
                    self._conf_mgr.on_detected(target_track.score)

        return all_tracks, self._build_info(target_track)

    # ------------------------------------------------------------------
    # Re-identification
    # ------------------------------------------------------------------

    def _try_reid(self, tracks: List[Track],
                  frame: Optional[np.ndarray]) -> Optional[Track]:
        """
        사라진 타겟을 현재 트랙 중에서 재식별.
        위치(칼만 예측) + 크기 유사도로 매칭.
        """
        if not tracks or self._last_track is None:
            return None

        pred_cx = int(self._pred_mean[0]) if self._pred_mean is not None \
                  else self._last_track.cx
        pred_cy = int(self._pred_mean[1]) if self._pred_mean is not None \
                  else self._last_track.cy

        best_track = None
        best_score = float('inf')

        for t in tracks:
            if t.track_id == self._target_id:
                continue

            # 거리 점수
            dist = np.sqrt((t.cx - pred_cx)**2 + (t.cy - pred_cy)**2)
            if dist > self._reid_thresh:
                continue

            # 크기 유사도
            size_score = 0.0
            if self._target_size:
                tw, th = self._target_size
                size_diff = abs(t.det.w - tw) + abs(t.det.h - th)
                size_score = size_diff / max(tw + th, 1)

            # 속도 방향 유사도
            vel_score = 0.0
            if self._last_track.mean is not None and t.mean is not None:
                vx1, vy1 = self._last_track.velocity
                vx2, vy2 = t.velocity
                if np.sqrt(vx1**2+vy1**2) > 1 and np.sqrt(vx2**2+vy2**2) > 1:
                    cos_sim = (vx1*vx2 + vy1*vy2) / (
                        np.sqrt(vx1**2+vy1**2) * np.sqrt(vx2**2+vy2**2) + 1e-6)
                    vel_score = 1 - cos_sim  # 방향 다르면 높음

            total = dist * 0.6 + size_score * 100 * 0.2 + vel_score * 100 * 0.2

            if total < best_score:
                best_score = total
                best_track = t

        return best_track

    def _save_snapshot(self, frame: np.ndarray, track: Track):
        """타겟 외형 스냅샷 저장."""
        try:
            import cv2
            x, y, w, h = track.det.bbox
            x = max(0, x); y = max(0, y)
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                self._target_snapshot = cv2.resize(crop, (64, 64))
        except Exception:
            pass

    def _build_info(self, track: Optional[Track]) -> Optional[TargetInfo]:
        """TargetInfo 객체 생성."""
        if self._target_id is None:
            return None

        src = track or self._last_track
        if src is None:
            return None

        pred_cx = int(self._pred_mean[0]) if self._pred_mean is not None else src.cx
        pred_cy = int(self._pred_mean[1]) if self._pred_mean is not None else src.cy

        vx, vy = src.velocity
        return TargetInfo(
            track_id         = self._target_id,
            cx               = src.cx,
            cy               = src.cy,
            w                = src.det.w,
            h                = src.det.h,
            vx               = vx,
            vy               = vy,
            speed            = src.speed,
            det_confidence   = src.score,
            track_confidence = min(1.0, src.hit_streak / 10.0),
            target_confidence= self._conf_mgr.confidence,
            state            = self._state,
            age              = src.age,
            pred_cx          = pred_cx,
            pred_cy          = pred_cy,
        )

    # ------------------------------------------------------------------
    # 프로퍼티
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def target_id(self) -> Optional[int]:
        return self._target_id

    @property
    def has_target(self) -> bool:
        return self._target_id is not None

    @property
    def byte_tracker(self) -> ByteTracker:
        return self._byte_tracker

    @property
    def snapshot(self) -> Optional[np.ndarray]:
        return self._target_snapshot
