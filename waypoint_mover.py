"""좌표 기반 웨이포인트 이동 모듈.

미리 지정한 좌표 리스트를 순서대로 Pico 클릭으로 이동합니다.
각 웨이포인트 도착 판정은 move_timeout_ms 경과 시 도착으로 간주.

RandomPatrolMover: 중심점 + 반경 기반 랜덤 순찰 (고정 좌표 없음)

사용법:
    mover = WaypointMover(
        waypoints=[
            {"x": 960, "y": 540, "label": "사냥터A", "wait_ms": 1500},
            {"x": 800, "y": 400, "label": "사냥터B", "wait_ms": 1000},
        ],
        move_timeout_ms=8000,
    )
    mover.start()
    while not mover.done:
        status = mover.tick(pico_worker)
        time.sleep(0.1)
"""

import logging
import random
import time

logger = logging.getLogger("waypoint_mover")


class WaypointMover:
    """웨이포인트 리스트를 순환하며 이동합니다."""

    def __init__(
        self,
        waypoints: list,
        move_timeout_ms: float = 5000.0,
        loop: bool = True,
        click_pulse_ms: int = 20,
    ):
        """
        Args:
            waypoints: [{"x","y","label"(optional),"wait_ms"(optional),
                         "clicks"(optional),"click_delay_ms"(optional)}, ...]
                wait_ms: 이 웨이포인트 도착 후 대기 시간 (기본 500ms)
                clicks: 클릭 횟수 (기본 1)
                click_delay_ms: 클릭 사이 딜레이 ms (기본 0)
            move_timeout_ms: 한 웨이포인트 이동 최대 대기 시간 (경과 시 도착 간주)
            loop: True면 마지막 웨이포인트 후 처음으로 순환
            click_pulse_ms: Pico 클릭 pulse 시간
        """
        if not waypoints:
            raise ValueError("waypoints가 비어 있습니다.")

        self.waypoints       = waypoints
        self.move_timeout_ms = move_timeout_ms
        self.loop            = loop
        self.click_pulse_ms  = click_pulse_ms

        self._idx          = 0
        self._state        = "IDLE"   # IDLE / MOVING / WAITING / DONE
        self._move_start_t = 0.0
        self._wait_until_t = 0.0

    # ── 공개 API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """순환 시작 (처음 웨이포인트로)."""
        self._idx   = 0
        self._state = "IDLE"
        logger.info(f"[WaypointMover] 시작: {len(self.waypoints)}개 웨이포인트")

    def reset(self) -> None:
        """처음으로 리셋."""
        self.start()

    @property
    def done(self) -> bool:
        """loop=False일 때 모든 웨이포인트 완료 여부."""
        return self._state == "DONE"

    @property
    def current_label(self) -> str:
        """현재 목표 웨이포인트 이름."""
        if self._idx < len(self.waypoints):
            return self.waypoints[self._idx].get("label", f"WP{self._idx}")
        return "DONE"

    @property
    def current_index(self) -> int:
        return self._idx

    def tick(self, pico_worker, frame=None) -> str:
        """매 루프마다 호출. 현재 상태를 반환합니다.

        Returns:
            "MOVING"   : 이동 중
            "ARRIVED"  : 방금 도착 (타임아웃)
            "WAITING"  : 도착 후 대기 중
            "DONE"     : 모든 웨이포인트 완료 (loop=False)
            "IDLE"     : 시작 전
        """
        now = time.time()

        # ── IDLE → 첫 웨이포인트로 이동 시작 ──────────────────────────
        if self._state == "IDLE":
            self._move_to_current(pico_worker, now)
            return "MOVING"

        # ── MOVING → 타임아웃 체크 ─────────────────────────────────────
        if self._state == "MOVING":
            elapsed_ms = (now - self._move_start_t) * 1000.0
            if elapsed_ms >= self.move_timeout_ms:
                logger.info(
                    f"[WaypointMover] '{self.current_label}' 도착 "
                    f"(타임아웃 {self.move_timeout_ms:.0f}ms)"
                )
                self._on_arrived(now)
                return "ARRIVED"
            return "MOVING"

        # ── WAITING → 대기 완료 후 다음 웨이포인트 ────────────────────
        if self._state == "WAITING":
            if now >= self._wait_until_t:
                self._advance(pico_worker, now)
            return "WAITING"

        # ── DONE ───────────────────────────────────────────────────────
        if self._state == "DONE":
            return "DONE"

        return self._state

    def force_next(self, pico_worker) -> None:
        """현재 웨이포인트를 건너뛰고 다음으로 강제 이동합니다."""
        logger.info(f"[WaypointMover] 강제 다음: '{self.current_label}' 스킵")
        self._advance(pico_worker, time.time())

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _move_to_current(self, pico_worker, now: float) -> None:
        wp          = self.waypoints[self._idx]
        ax          = wp["x"]
        ay          = wp["y"]
        label       = wp.get("label", f"WP{self._idx}")
        clicks      = wp.get("clicks", 1)
        click_delay = wp.get("click_delay_ms", 0) / 1000.0
        logger.info(
            f"[WaypointMover] → '{label}' ({ax},{ay}) "
            f"x{clicks} delay={click_delay:.1f}s"
        )
        print(f"[WaypointMover] → '{label}' ({ax},{ay}) x{clicks}")
        for i in range(clicks):
            if i > 0 and click_delay > 0:
                time.sleep(click_delay)
            pico_worker.click(ax, ay, self.click_pulse_ms)
        self._move_start_t = now
        self._state        = "MOVING"

    def _on_arrived(self, now: float) -> None:
        wp                 = self.waypoints[self._idx]
        wait_ms            = wp.get("wait_ms", 500)
        self._wait_until_t = now + wait_ms / 1000.0
        self._state        = "WAITING"

    def _advance(self, pico_worker, now: float) -> None:
        self._idx += 1
        if self._idx >= len(self.waypoints):
            if self.loop:
                self._idx = 0
                logger.info("[WaypointMover] 순환 반복 시작")
                self._move_to_current(pico_worker, now)
            else:
                logger.info("[WaypointMover] 모든 웨이포인트 완료")
                self._state = "DONE"
        else:
            self._move_to_current(pico_worker, now)


class RandomPatrolMover:
    """중심점 + 반경 기반 랜덤 순찰.

    고정 좌표 없이 center(x,y) 주변 radius 픽셀 내에서
    매번 다른 랜덤 목표로 이동합니다.

    config 예시:
        "patrol_mode": "random",
        "patrol_center": {"x": 900, "y": 400},
        "patrol_radius": 300,
        "patrol_interval_sec": 3.0

    사용:
        mover = RandomPatrolMover(center=(900,400), radius=300,
                                  move_timeout_ms=5000, interval_sec=3.0)
        mover.start()
        status = mover.tick(pico_worker)
        # status: "MOVING" / "ARRIVED" / "WAITING"
    """

    def __init__(
        self,
        center: tuple,          # (cx, cy) — 사냥터 중심 화면 좌표
        radius: int = 300,      # 랜덤 이동 반경 (px)
        move_timeout_ms: float = 5000.0,
        interval_sec: float = 3.0,   # 도착 후 다음 이동까지 대기
        click_pulse_ms: int = 20,
        screen_margin: int = 30,     # 화면 가장자리 최소 여백 (px)
        screen_w: int = 1920,
        screen_h: int = 1080,
    ):
        self._cx            = center[0]
        self._cy            = center[1]
        self._radius        = radius
        self._move_timeout  = move_timeout_ms
        self._interval      = interval_sec
        self._pulse         = click_pulse_ms
        self._margin        = screen_margin
        self._sw            = screen_w
        self._sh            = screen_h

        self._state         = "IDLE"   # IDLE / MOVING / WAITING
        self._move_start_t  = 0.0
        self._wait_until_t  = 0.0
        self._target        = (self._cx, self._cy)

    def start(self) -> None:
        self._state = "IDLE"

    def _next_target(self) -> tuple:
        """중심점 반경 내 랜덤 좌표 생성 (화면 경계 안쪽 보장)."""
        for _ in range(50):   # 최대 50회 시도 (반경 내 유효 좌표)
            angle = random.uniform(0, 2 * 3.14159)
            dist  = random.uniform(0, self._radius)
            tx = int(self._cx + dist * __import__("math").cos(angle))
            ty = int(self._cy + dist * __import__("math").sin(angle))
            # 화면 경계 체크
            if (self._margin <= tx <= self._sw - self._margin and
                    self._margin <= ty <= self._sh - self._margin):
                return tx, ty
        # 50회 실패 시 중심점 반환
        return self._cx, self._cy

    def tick(self, pico_worker) -> str:
        """매 루프마다 호출.

        Returns:
            "MOVING"  : 이동 중 (타임아웃 대기)
            "ARRIVED" : 방금 도착
            "WAITING" : 대기 중
        """
        now = time.time()

        # IDLE → 첫 랜덤 목표로 이동
        if self._state == "IDLE":
            self._move_to_next(pico_worker, now)
            return "MOVING"

        # MOVING → 타임아웃 체크
        if self._state == "MOVING":
            elapsed_ms = (now - self._move_start_t) * 1000.0
            if elapsed_ms >= self._move_timeout:
                tx, ty = self._target
                print(f"[RandomPatrol] 도착: ({tx},{ty})")
                self._wait_until_t = now + self._interval
                self._state = "WAITING"
                return "ARRIVED"
            return "MOVING"

        # WAITING → 대기 완료 후 다음 랜덤 목표
        if self._state == "WAITING":
            if now >= self._wait_until_t:
                self._move_to_next(pico_worker, now)
            return "WAITING"

        return self._state

    def _move_to_next(self, pico_worker, now: float) -> None:
        tx, ty = self._next_target()
        self._target       = (tx, ty)
        self._move_start_t = now
        self._state        = "MOVING"
        print(f"[RandomPatrol] → ({tx},{ty})  중심=({self._cx},{self._cy}) 반경={self._radius}")
        pico_worker.click(tx, ty, self._pulse)
