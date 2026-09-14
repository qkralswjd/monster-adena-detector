"""
adena_collector.py
------------------
몬스터 사망 후 해당 자리에 아데나(adena, class_id==1)가 생기면
자동으로 클릭해서 줍는 모듈.

흐름:
  1. 몬스터 사망 확정 시 → record_kill(cx, cy) 호출
     - 사망 위치와 탐색 반경을 기억
  2. 매 프레임 → check_and_collect(detections, ctrl) 호출
     - 기억된 위치 주변에서 adena(class_id==1) 탐색
     - 있으면 CLICK 으로 줍기
     - 없으면 collect_timeout_sec 후 포기 (아데나 없는 몬스터)

설정값 (config["adena"]):
  enabled          : bool  - 아데나 수집 기능 ON/OFF
  search_radius    : int   - 사망 위치 기준 탐색 반경 (픽셀)
  collect_timeout_sec : float - 탐색 대기 시간 (아데나 안 뜨면 포기)
  click_hold_ms    : int   - 줍기 클릭 유지 시간 (ms)
  max_picks        : int   - 한 사망 자리에서 최대 줍기 횟수
"""

import time
import math
from typing import Optional, List, Tuple
from detector import Detection


def _dist(ax: int, ay: int, bx: int, by: int) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


class AdenaCollector:

    def __init__(self,
                 enabled: bool = True,
                 search_radius: int = 120,
                 collect_timeout_sec: float = 3.0,
                 click_hold_ms: int = 50,
                 max_picks: int = 5,
                 mon_left: int = 0,
                 mon_top: int = 0):

        self._enabled            = enabled
        self._search_radius      = search_radius
        self._collect_timeout    = collect_timeout_sec
        self._click_hold_ms      = click_hold_ms
        self._max_picks          = max_picks
        self._mon_left           = mon_left
        self._mon_top            = mon_top

        # 현재 수집 대기 상태
        self._kill_x: Optional[int] = None   # 사망한 프레임 좌표 (절대)
        self._kill_y: Optional[int] = None
        self._kill_time: Optional[float] = None
        self._picks_done: int = 0             # 이번 사망 자리 줍기 횟수
        self._collecting: bool = False        # 수집 대기 중 여부

    # ─────────────────────────────────────────────────────────
    def record_kill(self, frame_cx: int, frame_cy: int):
        """
        몬스터 사망 확정 시 호출.
        frame_cx/cy: 사망한 몬스터의 프레임 내 중앙 좌표.
        """
        if not self._enabled:
            return

        # 절대좌표로 변환
        abs_x = frame_cx + self._mon_left
        abs_y = frame_cy + self._mon_top

        self._kill_x    = abs_x
        self._kill_y    = abs_y
        self._kill_time = time.time()
        self._picks_done = 0
        self._collecting = True

        print(f"[Adena] 사망 위치 기억: 프레임({frame_cx},{frame_cy}) → 절대({abs_x},{abs_y})")
        print(f"[Adena] 반경 {self._search_radius}px 내 아데나 탐색 시작 "
              f"(최대 {self._collect_timeout}초)")

    # ─────────────────────────────────────────────────────────
    def check_and_collect(self, detections: List[Detection], ctrl) -> bool:
        """
        매 프레임 호출.
        사망 자리 주변에 아데나 있으면 줍기 클릭.

        Returns:
            True  = 수집 완료 또는 타임아웃 (더 이상 대기 불필요)
            False = 아직 수집 중 or 비활성
        """
        if not self._enabled or not self._collecting:
            return False

        now = time.time()
        elapsed = now - self._kill_time

        # ── 타임아웃: 아데나 없는 몬스터 ──────────────────────
        if elapsed > self._collect_timeout:
            print(f"[Adena] 타임아웃({self._collect_timeout}초) → 아데나 없음, 다음 몬스터로")
            self._reset()
            return True

        # ── adena(class_id==1) 필터링 ─────────────────────────
        adenas = [d for d in detections if d.class_id == 1]
        if not adenas:
            # 아직 아데나 탐지 안 됨 → 계속 대기
            return False

        # ── 사망 위치 반경 내 아데나만 후보 ──────────────────
        kill_frame_x = self._kill_x - self._mon_left
        kill_frame_y = self._kill_y - self._mon_top

        nearby = [
            d for d in adenas
            if _dist(d.cx, d.cy, kill_frame_x, kill_frame_y) <= self._search_radius
        ]

        if not nearby:
            # 탐지된 아데나가 다른 위치 → 대기
            return False

        # ── 가장 가까운 아데나 줍기 ──────────────────────────
        target_adena = min(nearby,
                           key=lambda d: _dist(d.cx, d.cy, kill_frame_x, kill_frame_y))

        abs_x = target_adena.cx + self._mon_left
        abs_y = target_adena.cy + self._mon_top

        print(f"[Adena] 아데나 발견! 프레임({target_adena.cx},{target_adena.cy}) "
              f"→ 절대({abs_x},{abs_y})  conf={target_adena.confidence:.2f}")

        ctrl.click(abs_x, abs_y, hold_ms=self._click_hold_ms)
        self._picks_done += 1

        print(f"[Adena] 줍기 완료! ({self._picks_done}/{self._max_picks})")

        # ── 최대 줍기 횟수 도달 → 수집 종료 ─────────────────
        if self._picks_done >= self._max_picks:
            print(f"[Adena] 최대 줍기 횟수 도달 → 수집 종료")
            self._reset()
            return True

        return False

    # ─────────────────────────────────────────────────────────
    def _reset(self):
        """수집 상태 초기화."""
        self._kill_x     = None
        self._kill_y     = None
        self._kill_time  = None
        self._picks_done = 0
        self._collecting = False

    @property
    def is_collecting(self) -> bool:
        """현재 아데나 수집 대기 중인지."""
        return self._collecting

    @property
    def elapsed(self) -> float:
        """수집 대기 경과 시간(초). 대기 중 아닐 때는 0."""
        if self._kill_time is None:
            return 0.0
        return time.time() - self._kill_time
