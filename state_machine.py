"""자동 레벨링 메인 상태머신.

────────────────────────────────────────────────────────────
전체 흐름
────────────────────────────────────────────────────────────

    [IDLE]
       │ start() 호출
       ▼
    [ATTACKING_DUMMY]   ← 허수아비 드래그 공격 반복 (레벨 달성까지)
       │   HP < hp_threshold% → hp_potion_key 자동 사용
       │   OCR 레벨 >= target_level_dummy → MOVE_TO_HUNT_ZONE
       ▼
    [MOVE_TO_HUNT_ZONE]
       │ hunt_waypoints 순서대로 클릭 이동 (몬스터 감지 무시)
       │ 이동 중에도 HP 체크 → 물약
       │ 웨이포인트 전부 완주 후 → HUNTING
       ▼
    [HUNTING]
       │ YOLO 몬스터 탐지 + 공격
       │ patrol_waypoints 순찰 루프
       │ HP < hp_threshold% → 물약 자동 사용
       │ 아데나 감지 → LOOTING → 복귀
       │ OCR 레벨 >= target_level_hunt → DONE
       ▼
    [LOOTING]   ← 아데나 HSV 탐지 후 클릭
       │ 완료 → HUNTING 복귀
       ▼
    [DONE]  ← 완료

어느 상태에서든:
    stop() 호출 → IDLE
    HP < hp_threshold% → 물약 (쿨타임 체크)

────────────────────────────────────────────────────────────
"""

import logging
import math
import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from waypoint_mover  import WaypointMover, RandomPatrolMover
from loot_detector   import LootDetector
from level_detector  import LevelDetector
from recorder        import get_recorder

logger = logging.getLogger("state_machine")


class BotState(Enum):
    IDLE               = auto()
    ATTACKING_DUMMY    = auto()   # 허수아비 공격 (→ target_level_dummy)
    MOVE_TO_HUNT_ZONE  = auto()   # 사냥터 이동 (hunt_waypoints)
    HUNTING            = auto()   # 사냥터 사냥 + 순찰 (patrol_waypoints)
    LOOTING            = auto()   # 아데나 줍기
    DONE               = auto()   # 완료


class StateMachine:
    """자동 레벨링 전체 흐름 관리.

    config.json 구조 (level_detector / movement / dummy 섹션):
        dummy.drag_from      : 드래그 시작 {"x","y"}
        dummy.drag_to        : 드래그 끝   {"x","y"}
        dummy.drag_steps     : 드래그 분할 횟수 (기본 8)
        dummy.attack_interval_ms : 공격 간격 ms (기본 800)
        dummy.attack_duration_s  : 0=레벨달성까지, 양수=해당초 후 전환

        movement.hunt_waypoints  : 사냥터 이동 좌표 리스트
        movement.patrol_waypoints: 사냥터 내 순찰 좌표 리스트
        movement.move_timeout_ms : 웨이포인트 이동 타임아웃 ms

        level_detector.target_level_dummy : 허수아비 종료 레벨 (기본 5)
        level_detector.target_level_hunt  : 완료 레벨 (기본 10)
        level_detector.hp_threshold       : HP 물약 기준 % (기본 50.0)

        items.hp_potion_key       : HP 물약 키 (기본 "F4")
        items.potion_cooldown_sec : 물약 쿨타임 초 (기본 5.0)
    """

    def __init__(
        self,
        config: dict,
        ctrl,
        press_key_fn,
        capture_full_frame=None,   # 하위 호환용 (무시됨)
    ):
        """
        Args:
            config          : config.json 전체 dict
            ctrl            : PicoController or DummyController
            press_key_fn    : (key: str) -> None (키 입력 함수)
            capture_full_frame: 미사용 (하위 호환 유지)
        """
        self.cfg   = config
        self.ctrl  = ctrl
        self.press = press_key_fn

        self.state       = BotState.IDLE
        self._entered_at = time.time()

        self._load_config(config)

        logger.info("[StateMachine] 초기화 완료")

    def _load_config(self, cfg: dict) -> None:
        """config 로드 (reload_config 후 재호출 가능)."""
        dcfg  = cfg.get("dummy",          {})
        mcfg  = cfg.get("movement",       {})
        ldcfg = cfg.get("level_detector", {})
        icfg  = cfg.get("items",          {})

        # ── 허수아비 드래그 공격 설정 ────────────────────────────────────
        drag_from = dcfg.get("drag_from", {"x": 960, "y": 540})
        drag_to   = dcfg.get("drag_to",   {"x": 960, "y": 400})
        self.dummy_drag_from    = (drag_from.get("x", 960), drag_from.get("y", 540))
        self.dummy_drag_to      = (drag_to.get("x",   960), drag_to.get("y",   400))
        self.dummy_drag_steps   = dcfg.get("drag_steps", 8)
        self.dummy_atk_interval = dcfg.get("attack_interval_ms", 800) / 1000.0
        self.dummy_duration     = dcfg.get("attack_duration_s", 0.0)

        # ── 레벨/HP 설정 ─────────────────────────────────────────────────
        self.target_level_dummy = ldcfg.get("target_level_dummy", ldcfg.get("target_level", 5))
        self.target_level_hunt  = ldcfg.get("target_level_hunt",  10)
        self.hp_threshold       = ldcfg.get("hp_threshold", 50.0)
        level_roi               = ldcfg.get("level_roi")
        hp_roi                  = ldcfg.get("hp_roi")

        # ── 레벨/HP 감지기 ───────────────────────────────────────────────
        monitor = cfg.get("capture", {}).get("monitor", 1)
        self.level_det = LevelDetector(
            level_roi    = level_roi,
            hp_roi       = hp_roi,
            target_level = self.target_level_dummy,
            monitor      = monitor,
        )

        # ── 아이템 키 ────────────────────────────────────────────────────
        self.hp_potion_key     = icfg.get("hp_potion_key", "F4")
        self.potion_cooldown   = icfg.get("potion_cooldown_sec", 5.0)
        self._last_potion_t    = 0.0

        # ── 사냥터 이동 웨이포인트 (hunt_waypoints) ─────────────────────
        hunt_wps    = mcfg.get("hunt_waypoints",   [])
        patrol_wps  = mcfg.get("patrol_waypoints", [])
        wp_timeout  = mcfg.get("move_timeout_ms",  8000)

        if hunt_wps:
            self.hunt_mover = WaypointMover(
                waypoints       = hunt_wps,
                move_timeout_ms = wp_timeout,
                loop            = False,
            )
        else:
            self.hunt_mover = None

        # ── 순찰 모드 결정 (patrol_mode: "random" or "waypoint") ────────
        patrol_mode = mcfg.get("patrol_mode", "waypoint")

        if patrol_mode == "random":
            pc     = mcfg.get("patrol_center", {})
            pcx    = pc.get("x", 960)
            pcy    = pc.get("y", 540)
            pradius = mcfg.get("patrol_radius", 300)
            pinterval = mcfg.get("patrol_interval_sec", 3.0)
            self.patrol_mover = RandomPatrolMover(
                center          = (pcx, pcy),
                radius          = pradius,
                move_timeout_ms = wp_timeout,
                interval_sec    = pinterval,
            )
            print(f"[SM] 순찰 모드: 랜덤  중심=({pcx},{pcy})  반경={pradius}px  간격={pinterval}s")
        else:
            # 기존 고정 웨이포인트 순찰
            pwps = patrol_wps if patrol_wps else hunt_wps
            if pwps:
                self.patrol_mover = WaypointMover(
                    waypoints       = pwps,
                    move_timeout_ms = wp_timeout,
                    loop            = True,
                )
            else:
                self.patrol_mover = None

        # ── 아데나 탐지기 ────────────────────────────────────────────────
        adcfg = cfg.get("adena", {})
        scan_region = adcfg.get("scan_region", {"x": 360, "y": 240, "width": 1200, "height": 500})
        self.loot_detector = LootDetector(
            scan_region     = scan_region,
            scan_interval_s = adcfg.get("scan_interval_s", 0.3),
        )
        self.loot_click_delay = adcfg.get("click_delay_sec", 0.5)
        self.loot_timeout     = adcfg.get("check_timeout_sec", 4.0)

        # LOOTING 재진입 방지 파라미터
        self.loot_cooldown_sec      = adcfg.get("loot_cooldown_sec", 3.0)
        self.loot_revisit_radius_px = adcfg.get("loot_revisit_radius_px", 180)

        # ── 내부 상태 변수 ───────────────────────────────────────────────
        self._dummy_drag_done    = False
        self._last_dummy_atk     = 0.0
        self._patrol_started     = False
        self._loot_targets       = []
        self._loot_idx           = 0
        self._loot_start_t       = 0.0
        self._last_loot_click    = 0.0

        # LOOTING 재진입 방지용 상태변수
        self._last_loot_done_t   = 0.0   # 마지막 LOOTING 완료 시각
        self._last_loot_cx       = -1    # 마지막 루팅 아데나 평균 cx (-1=미설정)
        self._last_loot_cy       = -1    # 마지막 루팅 아데나 평균 cy (-1=미설정)
        # LOOTING 허용 조건: 최소 1회 이상 공격(combat_active=True)이 있었어야 함
        self._has_attacked_once  = False  # combat_active=True 전달된 적 있으면 True

        # ── 아이템 타이머 ────────────────────────────────────────────────
        speed1_key      = icfg.get("speed1_key", "F6")
        speed1_interval = icfg.get("speed1_interval_sec", 300)
        speed2_key      = icfg.get("speed2_key", "F7")
        speed2_interval = icfg.get("speed2_interval_sec", 480)
        self.speed1_key      = speed1_key
        self.speed1_interval = speed1_interval
        self.speed2_key      = speed2_key
        self.speed2_interval = speed2_interval
        self._last_speed1_t  = time.time()
        self._last_speed2_t  = time.time()

    def reload(self, config: dict) -> None:
        """config reload 후 재호출."""
        self.cfg = config
        self._load_config(config)

    # ── 공개 API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """자동 레벨링 시작 (허수아비 공격부터)."""
        logger.info("=" * 60)
        logger.info("[SM] ▶ 자동 레벨링 시작")
        logger.info(f"  허수아비 드래그: {self.dummy_drag_from} → {self.dummy_drag_to}")
        logger.info(f"  허수아비 목표 Lv.{self.target_level_dummy}  사냥터 목표 Lv.{self.target_level_hunt}")
        logger.info("=" * 60)
        self._enter(BotState.ATTACKING_DUMMY)

    def start_at_hunt_zone(self) -> None:
        """사냥터 이동부터 시작 (이미 허수아비 레벨 달성한 경우)."""
        logger.info("[SM] ▶ 사냥터 이동 시작")
        if self.hunt_mover:
            self.hunt_mover.start()
        self._enter(BotState.MOVE_TO_HUNT_ZONE)

    def start_hunting(self) -> None:
        """바로 HUNTING 시작 (이미 사냥터에 있는 경우)."""
        logger.info("[SM] ▶ 사냥터 사냥 즉시 시작")
        self._has_attacked_once = False  # 새 사냥 세션 시작 → 공격 전제 초기화
        self._enter(BotState.HUNTING)

    def stop(self) -> None:
        """중지."""
        logger.info("[SM] ■ 중지")
        self._enter(BotState.IDLE)

    @property
    def state_name(self) -> str:
        return self.state.name

    def update(self, frame, detections: list, combat_active: bool = False) -> None:
        """매 루프마다 호출.

        Args:
            frame         : ROI 기준 캡처 프레임 (BGR ndarray)
            detections    : YOLODetector.detect() 반환 Detection 리스트
            combat_active : 게임 자동사냥 진행 중 여부 (main.py 관리).
                            True이면 순찰·LOOTING 전환을 차단한다.

        Note:
            레벨/HP 인식은 LevelDetector 내부에서 ROI 직접 캡처.
            전체화면 캡처 불필요.
        """
        if self.state in (BotState.IDLE, BotState.DONE):
            return

        # ── 아이템 공통 처리 ──────────────────────────────────────────
        self._check_hp_and_use_potion()
        self._check_speed_items()

        # ── 상태별 처리 ───────────────────────────────────────────────
        if self.state == BotState.ATTACKING_DUMMY:
            self._update_attacking_dummy()

        elif self.state == BotState.MOVE_TO_HUNT_ZONE:
            self._update_move_to_hunt_zone(detections)

        elif self.state == BotState.HUNTING:
            # combat_active=True가 한 번이라도 전달되면 기록
            if combat_active and not self._has_attacked_once:
                self._has_attacked_once = True
                print("[SM] 첫 공격 확인 → LOOTING 허용 상태로 전환")
            self._update_hunting(detections, combat_active)

        elif self.state == BotState.LOOTING:
            self._update_looting()

    # ── HP / 아이템 공통 ──────────────────────────────────────────────────

    def _check_hp_and_use_potion(self) -> None:
        hp = self.level_det.read_hp()
        if hp is None:
            return
        now = time.time()
        if hp < self.hp_threshold and now - self._last_potion_t >= self.potion_cooldown:
            print(f"[Item] HP {hp:.1f}% < {self.hp_threshold:.0f}% → {self.hp_potion_key} 사용")
            self.press(self.hp_potion_key)
            self._last_potion_t = now

    def _check_speed_items(self) -> None:
        now = time.time()
        if now - self._last_speed1_t >= self.speed1_interval:
            print(f"[Item] {self.speed1_key} 이속아이템 사용")
            self.press(self.speed1_key)
            self._last_speed1_t = now
        if now - self._last_speed2_t >= self.speed2_interval:
            print(f"[Item] {self.speed2_key} 이속아이템 사용")
            self.press(self.speed2_key)
            self._last_speed2_t = now

    # ── ATTACKING_DUMMY ───────────────────────────────────────────────────

    def _update_attacking_dummy(self) -> None:
        now = time.time()

        # 레벨 체크
        level = self.level_det.read_level()
        if level is not None and level >= self.target_level_dummy:
            print(f"[SM] Lv.{level} 달성! → 사냥터 이동 시작")
            if self.hunt_mover:
                self.hunt_mover.start()
            self._enter(BotState.MOVE_TO_HUNT_ZONE)
            return

        # 시간 기반 종료
        if self.dummy_duration > 0:
            if now - self._entered_at >= self.dummy_duration:
                print(f"[SM] 허수아비 {self.dummy_duration:.0f}초 완료 → 사냥터 이동")
                if self.hunt_mover:
                    self.hunt_mover.start()
                self._enter(BotState.MOVE_TO_HUNT_ZONE)
                return

        # 드래그 공격 (1회 실행 — 게임이 자동 반복)
        if not self._dummy_drag_done:
            fx, fy = self.dummy_drag_from
            tx, ty = self.dummy_drag_to
            self.ctrl.drag_attack(fx, fy, hold_ms=80)
            self._dummy_drag_done = True
            lv_str = str(level) if level is not None else "?"
            print(
                f"[Dummy] 드래그 공격 ({fx},{fy})→({tx},{ty}) "
                f"Lv={lv_str}/{self.target_level_dummy}"
            )

        # 추가 반복 공격 (attack_interval_ms마다)
        elif now - self._last_dummy_atk >= self.dummy_atk_interval:
            fx, fy = self.dummy_drag_from
            tx, ty = self.dummy_drag_to
            self.ctrl.drag_attack(fx, fy, hold_ms=80)
            self._last_dummy_atk = now

    # ── MOVE_TO_HUNT_ZONE ─────────────────────────────────────────────────

    def _update_move_to_hunt_zone(self, detections: list) -> None:
        # 이동 중 몬스터 감지 무시 — 웨이포인트 완주 후 HUNTING 전환
        if self.hunt_mover is None:
            print("[SM] hunt_waypoints 없음 → HUNTING")
            self._enter(BotState.HUNTING)
            return

        status = self.hunt_mover.tick(self.ctrl)
        if status == "ARRIVED":
            label = self.hunt_mover.current_label
            print(f"[SM] '{label}' 도착 — 대기 중")
        elif status == "DONE":
            print("[SM] 사냥터 도착 완료 → HUNTING")
            self._enter(BotState.HUNTING)

    # ── HUNTING ───────────────────────────────────────────────────────────

    def _update_hunting(self, detections: list, combat_active: bool = False) -> None:
        now = time.time()

        # 최초 진입 시에만 순찰 시작.
        # LOOTING 복귀 등 재진입(HUNTING → LOOTING → HUNTING) 시에는
        # _patrol_started=True를 유지하여 patrol_mover를 reset하지 않음.
        # (_enter(HUNTING)에서 _patrol_started를 False로 초기화하지 않도록 변경됨)
        if not self._patrol_started:
            if self.patrol_mover:
                self.patrol_mover.start()
            self._patrol_started = True
            print("[SM] 순찰 시작 (patrol_waypoints loop)")
            get_recorder().log_event("PATROL_START")

        # 레벨 체크
        level = self.level_det.read_level()
        if level is not None and level >= self.target_level_hunt:
            print(f"[SM] Lv.{level} 달성! → 완료")
            self._enter(BotState.DONE)
            return

        monsters = [d for d in detections if d.class_id == 0]
        adenas   = [d for d in detections if d.class_id == 1]

        # ── 전투 중이면 순찰/LOOTING 전환 모두 차단 ──────────────────────
        # combat_active=True: drag_attack() 이후 게임 자동사냥이 진행 중.
        # monsters=[] 이어도 일시적 YOLO miss일 수 있으므로 전투 유지.
        # 전투 종료 판정은 main.py에서 tracker 소실 신호로 수행.
        if combat_active:
            if not monsters and not adenas:
                pass  # YOLO miss — 전투 중으로 간주, 아무것도 하지 않음
            # monsters가 보이거나 adenas가 보여도 combat_active=True이면 차단
            return  # 순찰/LOOTING 로직에 도달하지 않음

        if monsters:
            # 몬스터 있으면 공격 (main.py의 tracker/attack 로직이 처리)
            pass
        else:
            # 몬스터 없을 때만 아데나 탐지 & 순찰 이동
            if adenas:
                self._trigger_looting(adenas, now)
                if self.state == BotState.LOOTING:
                    return

            # 순찰 이동 (combat_active=False 확인됨 → 전투 중 차단 없음)
            if self.patrol_mover:
                status = self.patrol_mover.tick(self.ctrl)
                if status == "ARRIVED":
                    # RandomPatrolMover는 current_label 없음
                    label = getattr(self.patrol_mover, "current_label", "랜덤")
                    print(f"[SM] 순찰 '{label}' 도착")

    def _trigger_looting(self, adenas: list, now: float) -> None:
        """아데나 감지 시 LOOTING 진입 여부를 판단하고 진입.

        LOOTING 재진입 방지 로직:
          1. 루팅 완료 후 loot_cooldown_sec 이내  → 무시
          2. 쿨타임 지났어도 이전 루팅 좌표 반경
             loot_revisit_radius_px 이내         → 같은 아데나로 판단, 무시
          3. 위 조건에 해당 안 되면               → LOOTING 진입
        """
        # ── 0. 공격 전제 조건 체크 ──────────────────────────────────
        # 최소 1회 이상 공격(combat_active=True)이 발생한 후에만 LOOTING 허용.
        # 공격 없이 화면에 보이는 기존 아데나 때문에 LOOTING 진입하는 것을 방지.
        if not self._has_attacked_once:
            print("[SM] 아데나 감지됐으나 아직 공격 전 → LOOTING 차단")
            return

        # ── 1. 쿨타임 체크 ──────────────────────────────────────────
        elapsed_since_loot = now - self._last_loot_done_t
        if self._last_loot_done_t > 0 and elapsed_since_loot < self.loot_cooldown_sec:
            remaining = self.loot_cooldown_sec - elapsed_since_loot
            print(f"[SM] 아데나 감지됐으나 쿨타임 중 → 무시 "
                  f"({remaining:.1f}s 남음)")
            return

        # ── 2. 좌표 비교 (쿨타임 지난 후에도 같은 위치 재감지 방지) ─
        if self._last_loot_cx >= 0:
            adena_cx = sum(d.cx for d in adenas) // len(adenas)
            adena_cy = sum(d.cy for d in adenas) // len(adenas)
            dist = math.hypot(adena_cx - self._last_loot_cx,
                              adena_cy - self._last_loot_cy)
            if dist < self.loot_revisit_radius_px:
                print(f"[SM] 아데나 감지됐으나 직전 루팅 위치와 동일한 것으로 판단 → 무시 "
                      f"(dist={dist:.0f}px < {self.loot_revisit_radius_px}px, "
                      f"기준=({self._last_loot_cx},{self._last_loot_cy}))")
                return

        # ── 3. LOOTING 진입 ──────────────────────────────────────────
        self._loot_targets = [(d.cx, d.cy) for d in adenas]
        self._loot_idx     = 0
        self._loot_start_t = now
        self.loot_detector.invalidate()
        print(f"[SM] 아데나 {len(adenas)}개 감지 → LOOTING")
        get_recorder().log_event("LOOT_START", {
            "count":   len(adenas),
            "targets": [(d.cx, d.cy) for d in adenas],
        })
        self._enter(BotState.LOOTING)

    # ── LOOTING ───────────────────────────────────────────────────────────

    def _update_looting(self) -> None:
        now = time.time()

        # 타임아웃
        if now - self._loot_start_t >= self.loot_timeout:
            print("[SM] 아데나 줍기 타임아웃 → HUNTING 복귀")
            print("[LOOT] DONE (타임아웃)")
            get_recorder().log_event("LOOT_DONE", {"reason": "timeout"})
            self._enter(BotState.HUNTING)
            return

        if not self._loot_targets or self._loot_idx >= len(self._loot_targets):
            print("[SM] 아데나 줍기 완료 → HUNTING 복귀")
            print("[LOOT] DONE")
            get_recorder().log_event("LOOT_DONE", {"reason": "all_clicked"})
            self._enter(BotState.HUNTING)
            return

        if now - self._last_loot_click >= self.loot_click_delay:
            cx, cy = self._loot_targets[self._loot_idx]
            self.ctrl.click(int(cx), int(cy))
            print(f"[Loot] 줍기 클릭 → ({cx},{cy}) {self._loot_idx+1}/{len(self._loot_targets)}")
            get_recorder().log_event("LOOT_CLICK", {
                "cx":    int(cx),
                "cy":    int(cy),
                "index": self._loot_idx + 1,
                "total": len(self._loot_targets),
            })
            self._last_loot_click = now
            self._loot_idx += 1

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _enter(self, new_state: BotState) -> None:
        if self.state != new_state:
            print(f"[SM] {self.state.name} → {new_state.name}")
            logger.info(f"[SM] {self.state.name} → {new_state.name}")
        self.state        = new_state
        self._entered_at  = time.time()

        # 상태별 초기화
        if new_state == BotState.ATTACKING_DUMMY:
            self._dummy_drag_done = False
            self._last_dummy_atk  = 0.0

        elif new_state == BotState.HUNTING:
            # _patrol_started 는 여기서 초기화하지 않음.
            # 최초 진입(IDLE/MOVE_TO_HUNT_ZONE → HUNTING)에서만 patrol_mover.start()를
            # 호출하고, LOOTING → HUNTING 복귀 시에는 재시작하지 않는다.
            # 최초 진입 판정: _patrol_started 값이 False인 상태에서 진입 → 그때만 start().
            # 이를 위해 LOOTING → HUNTING 재진입 시 _patrol_started를 False로 바꾸지 않음.
            # 주의: start_hunting() / start_at_hunt_zone() 호출 직후에는
            #        _load_config()가 _patrol_started=False로 초기화하므로
            #        최초 진입 시 patrol_mover.start()가 정상 호출됨.
            if self.state not in (BotState.LOOTING,):
                # LOOTING 복귀가 아닌 경우(최초 진입, MOVE_TO_HUNT_ZONE → HUNTING 등)는
                # patrol_started를 False로 세팅해 patrol_mover.start() 호출하도록 함.
                self._patrol_started = False
            else:
                # LOOTING → HUNTING 복귀: patrol_mover 상태 유지 (재시작 안 함)
                # 루팅 완료 시각 및 좌표 기록 (LOOTING 재진입 방지용)
                self._last_loot_done_t = time.time()
                if self._loot_targets:
                    xs = [c[0] for c in self._loot_targets]
                    ys = [c[1] for c in self._loot_targets]
                    self._last_loot_cx = sum(xs) // len(xs)
                    self._last_loot_cy = sum(ys) // len(ys)
                    print(f"[HUNT] RESUME (순찰 상태 유지) "
                          f"루팅좌표=({self._last_loot_cx},{self._last_loot_cy}) "
                          f"쿨타임={self.loot_cooldown_sec}s")
                    get_recorder().log_event("HUNT_RESUME", {
                        "loot_cx":      self._last_loot_cx,
                        "loot_cy":      self._last_loot_cy,
                        "cooldown_sec": self.loot_cooldown_sec,
                    })
                else:
                    print("[HUNT] RESUME (순찰 상태 유지)")
                    get_recorder().log_event("HUNT_RESUME")

        elif new_state == BotState.LOOTING:
            print("[LOOT] START (SM 진입)")

        elif new_state == BotState.DONE:
            print("[SM] ✅ 레벨링 완료!")
            logger.info("[SM] 레벨링 완료")
