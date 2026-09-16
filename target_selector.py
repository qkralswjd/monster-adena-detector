"""
target_selector.py
------------------
타겟 선택 + 추적 로직.

동작 원칙:
  1. 타겟 없음 → cy >= MIN_TARGET_CY 조건의 몬스터 중 confidence 최고값 선택
  2. 타겟 고정 후 → 속도 벡터 + ghost 예측 위치 기반 추적
     - 연속 탐지(miss 없는 프레임): dt(프레임 간격)로 순간 속도 갱신
     - miss 후 재연결:              elapsed(마지막 탐지 이후)로 평균 속도 계산
     - miss 중:                     ghost(예측 위치)를 ref로 사용해 재연결 탐색
     - max_allowed_dist = max(max_speed * elapsed, min_search_r)
       → 속도 미확보 상태에서도 min_search_r(60px) 보장
  3. miss_timeout 초 이상 탐지 안 되면 → TARGET_LOST (사망 추정, 가림/이탈과 구분 불가) → 새 타겟 선택

변수 역할 분리:
  dt      : 직전 update() ~ 현재 update() 사이의 실제 경과시간
            → 연속 탐지 시 순간 속도 계산에 사용
            → 0 < dt <= DT_MAX(0.2s) 로 클램핑해 비정상 값 방어
  elapsed : _last_seen(마지막 실제 탐지/재연결) 이후 경과시간
            → ghost 예측 범위 계산·재연결 탐색 반경에 사용
  elapsed_since_continuous : _last_seen_continuous(마지막 연속탐지 성공) 이후 경과시간
            → miss_timeout 판정 전용
            → 재연결 성공(was_continuous=False)으로는 갱신되지 않음

ghost 용도:
  - 오버레이 표시: miss 중 박스가 이동 방향으로 따라감
  - 재연결 탐색 ref: 예측 위치 주변에서 탐색
  - 공격 좌표에는 절대 사용하지 않음
    재연결 성공 시 실제 탐지된 Detection 반환 → miss_elapsed=0.0 → 공격 가능
    miss 중에는 elapsed > 0.0 → main.py 공격 조건 미충족 → 공격 안 됨

좌표 원칙:
  - 입력/출력 모두 ROI 기준 좌표 그대로. 변환 없음.
"""

import math
import time
import copy
from typing import List, Optional, Tuple
from detector import Detection

# 신규 타겟 선택 시 최소 cy 필터.
# 공격 판정(main.py min_cy)과 동일 값 유지.
# 이 값 미만인 몬스터는 타겟 선택 단계에서 원천 차단.
MIN_TARGET_CY: int = 150

# 신규 타겟 선택 시 최소 confidence 필터 기본값.
# 실제 사용값은 config.json → target.min_conf 이며,
# TargetTracker(min_conf=...) 로 전달돼 인스턴스별로 관리된다.
# 이 상수는 TargetTracker 생성 시 min_conf가 생략된 경우의 fallback 전용.
# main.py가 항상 config 값을 명시적으로 전달하므로 실질적으로는 사용되지 않음.
_DEFAULT_MIN_CONF: float = 0.20

# 재연결 탐색 최소 보장 반경 (px).
# 속도 미확보 상태(초기, vx=vy=0)에서 max_speed * elapsed가
# 너무 작아질 때 최소한의 탐색 범위를 보장.
# 60px 근거: 로그상 재연결 성공 케이스 최소 이동량 46px + 여유.
# 이 값이 "동일 몬스터 보장 거리"가 아님.
# 후보가 여러 개일 때는 ref와의 거리가 가장 작은 것을 선택함.
_MIN_SEARCH_R: float = 60.0

# dt 비정상 상한 (초).
# 슬립·디버거 중단 등으로 dt가 비정상적으로 클 때 속도가 0에 수렴하는 것을 방지.
# 0.2s(약 5fps) 초과분은 클램핑. 이 이상이면 연속탐지로 보지 않음.
_DT_MAX: float = 0.2


def select_by_confidence(
        detections: List[Detection],
        min_conf: float = _DEFAULT_MIN_CONF,
) -> Optional[Detection]:
    """
    다음 3가지 조건을 모두 만족하는 몬스터 중 confidence 최고값 반환.
      1. class_id == 0  (monster)
      2. cy >= MIN_TARGET_CY(150)  — 화면 상단 오탐 차단
      3. confidence >= min_conf    — config.json target.min_conf 기준

    [이번 변경]
      confidence >= min_conf 조건 추가.
      min_conf는 TargetTracker(min_conf=...) 를 통해 config.json 값이 전달됨.
      → Tracker 타겟 선택 단계와 main.py 공격 조건이 동일한 기준(min_conf)을 공유.
      → conf < min_conf 인 탐지는 Tracker가 타겟으로 선택하지 않음.
        (이전: Tracker가 선택 → 3.5s 추적 → 공격 0건 → TARGET_LOST 낭비)

    [유지하는 것]
      - cy >= MIN_TARGET_CY 필터
      - max(confidence) 선택 방식
      - cx 경계 필터 없음
      - 중앙 거리 가중치 없음
    """
    monsters = [
        d for d in detections
        if d.class_id == 0
        and d.cy >= MIN_TARGET_CY
        and d.confidence >= min_conf
    ]
    if not monsters:
        return None
    return max(monsters, key=lambda d: d.confidence)


def _make_ghost(src: Detection, cx: int, cy: int) -> Detection:
    """src Detection 기반으로 cx, cy 위치만 다른 예측 Detection 생성."""
    g = copy.copy(src)
    g.x = cx - src.w // 2
    g.y = cy - src.h // 2
    return g


class TargetTracker:
    """
    타겟 1개 고정 + 속도 벡터 기반 추적.

    핵심 변경 요약:
    ─────────────────────────────────────────────────────
    [변경 1] ghost 계산 조건 제거 (vx=0이어도 항상 계산)
      전: if elapsed > 0 and (self._vx != 0 or self._vy != 0): ghost = ...
          else: ghost = None  → 초기 상태에서 ref = 마지막 위치 고정
      후: 항상 ghost 계산. vx=vy=0이면 pred = 마지막 위치 그대로.
          → 항상 ref = ghost → 탐색 기준점 통일

    [변경 2] max_allowed_dist 공식
      전: max_speed * max(elapsed, 0.05)
          → 30fps 첫 프레임: 250 * 0.05 = 12.5px (하한 클램핑)
          → 몬스터가 13px만 이동해도 재연결 실패
      후: max(max_speed * elapsed, min_search_r=60)
          → 30fps 첫 프레임: max(250*0.033, 60) = max(8.25, 60) = 60px
          → 속도 미확보 상태에서도 60px 탐색 범위 보장
          → elapsed 누적 시 동적 반경이 커지며 min_search_r을 자연스럽게 추월

    [변경 3] dt / elapsed 역할 분리 + dt 방어 클램핑
      dt      = 직전 update() 호출 간격 → 연속탐지 속도 계산
      elapsed = 마지막 실제 탐지 이후   → ghost 범위·재연결 반경
      전: dt가 선언만 되고 사용 안 됨 (dead variable)
      후: dt로 연속탐지 순간속도 계산. dt > DT_MAX(0.2s)이면 클램핑.

    [변경 5] miss_timeout 기준점 분리 (_last_seen_continuous)
      전: _last_seen = now (재연결 성공 시도 갱신)
          → miss 후 재연결이 반복되면 타이머가 계속 리셋됨
          → TARGET_LOST가 10초 이상 지연되는 원인
      후: _last_seen_continuous = now (연속탐지 성공 시에만 갱신)
          _last_seen = now (이전과 동일, ghost 범위 계산용 유지)
          miss_timeout 판정: elapsed_since_continuous > miss_timeout
          → 재연결이 아무리 반복돼도 기준시각은 마지막 연속탐지로 고정
          → 연속탐지 기준 3.5s 후 TARGET_LOST 보장

    [변경 4] 연속탐지 / miss후재연결 속도 계산 분기
      연속탐지(elapsed < dt*2): vx = Δcx / dt   → 순간 속도
      miss후재연결:             vx = Δcx / elapsed → 평균 속도

    유지하는 것:
      - miss_timeout = 3.5s
      - 재연결 성공: return self._target, 0.0   (실제 Detection 반환)
      - miss 중:     return self.target, elapsed (ghost 반환, miss_elapsed>0)
      - miss_elapsed == 0.0 조건 → main.py 공격 조건 변경 없음
      - ghost는 오버레이·재연결 ref 용도만. 공격 좌표 사용 없음.
      - EMA / 복합점수 / confidence 가중치 없음
      - 추가 상태변수: _min_search_r 1개만
    ─────────────────────────────────────────────────────
    """

    def __init__(self,
                 miss_timeout: float = 3.5,
                 max_dist: float = 150,            # 하위 호환용 (현재 미사용)
                 max_speed: float = 250,            # px/초
                 min_search_r: float = _MIN_SEARCH_R,
                 min_conf: float = _DEFAULT_MIN_CONF):  # config.json target.min_conf
        self._target       = None         # 현재 추적 중인 실제 Detection
        self._ghost        = None         # miss 중 예측 위치 Detection (오버레이·ref용)

        # ── 시각 추적 변수 ────────────────────────────────────────────────
        # _last_seen            : 마지막 실제 탐지 OR 재연결 성공 시각
        #                         ghost 예측 좌표 계산 및 재연결 탐색 반경에 사용
        #                         (재연결 성공 시에도 갱신 → ghost가 실제 위치로 리셋)
        self._last_seen             = 0.0

        # _last_seen_continuous : 마지막 연속탐지(was_continuous=True) 성공 시각
        #                         miss_timeout 판정 전용.
        #                         재연결 성공(was_continuous=False)으로는 갱신되지 않음.
        #                         → 재연결이 반복돼도 타이머 기준점이 리셋되지 않음
        self._last_seen_continuous  = 0.0

        self._miss_timeout = miss_timeout
        self._max_speed    = max_speed
        self._min_search_r = min_search_r
        # config.json → target.min_conf 에서 전달받은 값.
        # select_by_confidence() 호출 시 전달해 Tracker 선택 단계에 적용.
        self._min_conf     = min_conf

        # 속도 벡터 (px/초). 항상 max_speed 이하로 클램핑됨.
        self._vx = 0.0
        self._vy = 0.0

        # dt 계산용: 직전 update() 호출 시각
        self._prev_t = 0.0

    @property
    def target(self) -> Optional[Detection]:
        """현재 타겟. miss 중이면 ghost 반환 (오버레이용)."""
        return self._ghost if self._ghost is not None else self._target

    def update(self, detections: List[Detection]) -> Tuple[Optional[Detection], float]:
        """
        매 프레임 호출.

        반환값:
          (Detection, miss_elapsed)
          - miss_elapsed == 0.0 : 이번 프레임에 실제 탐지 성공 → 공격 가능
          - miss_elapsed  > 0.0 : miss 중 → 공격 불가

        공격 좌표는 항상 실제 탐지된 self._target의 cx/cy.
        ghost.cx/cy는 공격 좌표에 절대 사용하지 않음.
        """
        monsters = [d for d in detections if d.class_id == 0]
        now = time.time()

        # ── dt: 직전 프레임과의 실제 경과시간 ────────────────────────
        # 연속 탐지 시 순간 속도 계산에만 사용.
        # _prev_t == 0.0 이면 첫 호출 → 기본값 0.033s(30fps 가정).
        # 비정상 큰 dt (슬립·중단 등) 방어: _DT_MAX(0.2s)로 클램핑.
        if self._prev_t > 0:
            raw_dt = now - self._prev_t
            dt = min(max(raw_dt, 1e-6), _DT_MAX)  # 0 < dt <= 0.2s
        else:
            dt = 0.033  # 첫 호출 기본값
        self._prev_t = now

        # ── 타겟 없음 → 새 타겟 선택 ─────────────────────────────────
        if self._target is None:
            # min_conf 전달: cy>=MIN_TARGET_CY AND conf>=self._min_conf 동시 적용
            new = select_by_confidence(monsters, min_conf=self._min_conf)
            if new:
                self._target               = new
                self._ghost                = None
                self._last_seen            = now  # ghost 범위 기준 갱신
                self._last_seen_continuous = now  # miss_timeout 기준 갱신
                self._vx = self._vy = 0.0
                print(f"[Tracker] ★ 타겟 선택: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self.target, 0.0

        # ── elapsed: 마지막 실제 탐지/재연결 이후 경과시간 ────────────
        # ghost 예측 좌표·재연결 탐색 반경 계산에 사용.
        # (재연결 성공 시에도 _last_seen이 갱신되므로 ghost가 실제 위치로 리셋됨)
        elapsed = now - self._last_seen

        # ── elapsed_since_continuous: 마지막 연속탐지 이후 경과시간 ─────
        # miss_timeout 판정 전용.
        # was_continuous=False인 재연결 성공으로는 갱신되지 않으므로
        # 재연결이 반복돼도 이 값은 계속 증가한다.
        elapsed_since_continuous = now - self._last_seen_continuous

        # ── ghost(예측 위치) 계산 ─────────────────────────────────────
        # [변경 전]
        #   if elapsed > 0 and (self._vx != 0 or self._vy != 0):
        #       ghost = 예측 위치
        #   else:
        #       ghost = None  ← vx=vy=0이면 ghost 없음 → ref = 마지막 위치 고정
        #
        # [변경 후]
        #   항상 ghost 계산. vx=vy=0이면 pred = 마지막 위치 그대로.
        #   → ghost가 항상 존재 → ref = ghost 경로로 통일
        #   → 초기 상태에서도 ref가 유효하게 동작
        pred_cx = int(self._target.cx + self._vx * elapsed)
        pred_cy = int(self._target.cy + self._vy * elapsed)
        pred_cx = max(0, pred_cx)
        pred_cy = max(0, pred_cy)
        self._ghost = _make_ghost(self._target, pred_cx, pred_cy)

        # 탐색 기준점: 항상 ghost (vx=vy=0이면 = 마지막 위치)
        ref = self._ghost

        # ── 재연결 탐색 반경 ──────────────────────────────────────────
        # [변경 전]
        #   max_speed * max(elapsed, 0.05)
        #   → 30fps 첫 프레임: 250 * 0.05 = 12.5px (하한 클램핑)
        #   → 몬스터가 13px만 이동해도 재연결 실패
        #
        # [변경 후]
        #   max(max_speed * elapsed, min_search_r)
        #   → 30fps 첫 프레임: max(250*0.033, 60) = max(8.25, 60) = 60px
        #   → 속도 미확보 상태에서도 60px 보장
        #   → elapsed가 누적될수록 max_speed*elapsed가 min_search_r을 추월
        #      elapsed > 0.24s → 250*0.24 = 60px → 이후 동적 반경 주도
        max_allowed_dist = max(
            self._max_speed * elapsed,  # 동적 반경 (elapsed 그대로, 클램핑 없음)
            self._min_search_r          # 최소 보장 반경 (60px)
        )

        # ── 가장 가까운 몬스터 탐색 (ref 기준) ───────────────────────
        # best_dist를 max_allowed_dist로 초기화해서
        # 반경 밖은 후보에서 제외하고, 반경 안에서 ref와 가장 가까운 것을 선택.
        # confidence >= min_conf 조건 추가:
        #   재연결 탐색에서도 동일한 confidence 기준 적용.
        #   이전: confidence 무관하게 거리만으로 재연결 → conf=0.18도 재연결됨
        #   이후: conf < min_conf 후보는 재연결 탐색에서도 건너뜀
        best      = None
        best_dist = max_allowed_dist

        for d in monsters:
            if d.confidence < self._min_conf:   # confidence 기준 미달 → 재연결 불가
                continue
            dist = math.hypot(d.cx - ref.cx, d.cy - ref.cy)
            if dist < best_dist:
                best_dist = dist
                best = d

        # ── 재연결 성공 ───────────────────────────────────────────────
        if best is not None:
            prev_cx, prev_cy = self._target.cx, self._target.cy

            # 속도 계산: 연속탐지 vs miss후재연결 분기
            #
            # 연속탐지 판별: elapsed < dt * 2.0
            #   = 마지막 탐지가 1~2프레임 이내 → miss 없이 이어진 것으로 판단
            #   → dt(실제 프레임 간격)로 순간 속도 계산
            #
            # miss후재연결: elapsed >= dt * 2.0
            #   = 1프레임 이상 miss 후 재연결
            #   → elapsed(마지막탐지~현재)로 평균 변위 속도 계산
            #   → "순간 속도"가 아니라 "평균 속도"임에 유의
            was_continuous = (elapsed < dt * 2.0)
            time_divisor   = dt if was_continuous else elapsed

            # time_divisor는 위 분기에서 dt(0 < dt <= DT_MAX) 또는
            # elapsed(>= dt*2 > 0)이므로 항상 양수. 추가 0 체크 생략 가능하나
            # 방어를 위해 유지.
            if time_divisor > 0:
                new_vx = (best.cx - prev_cx) / time_divisor
                new_vy = (best.cy - prev_cy) / time_divisor

                # 속도 크기 클램핑 (방향 유지, 크기만 max_speed로 제한)
                speed = math.hypot(new_vx, new_vy)
                if speed > self._max_speed:
                    scale = self._max_speed / speed
                    new_vx *= scale
                    new_vy *= scale

                self._vx = new_vx
                self._vy = new_vy

            self._target    = best
            self._ghost     = None   # 실제 탐지 확보 → ghost 제거
            self._last_seen = now    # ghost 기준 갱신 (연속/재연결 모두 갱신)

            # ── miss_timeout 기준점 갱신: 연속탐지 성공 시에만 ───────────
            # was_continuous=True  : 정상 연속탐지 → 기준시각 갱신
            # was_continuous=False : miss 후 재연결 → 기준시각 갱신 금지
            #   이유: 재연결이 반복될 때마다 갱신하면 3.5s 타이머가 계속 리셋되어
            #         TARGET_LOST가 10초 이상 지연되는 버그가 발생함.
            if was_continuous:
                self._last_seen_continuous = now

            moved = math.hypot(best.cx - prev_cx, best.cy - prev_cy)
            if moved > 20:
                mode = "연속" if was_continuous else f"miss후({elapsed:.2f}s)"
                print(f"[Tracker] 이동({mode}): ({prev_cx},{prev_cy}) → "
                      f"({best.cx},{best.cy})  Δ{moved:.0f}px  "
                      f"vx={self._vx:.0f} vy={self._vy:.0f}")

            # 실제 탐지된 Detection 반환 + miss_elapsed=0.0 → 공격 가능
            return self._target, 0.0

        # ── 탐지 실패 → miss 타이머 체크 ─────────────────────────────
        # miss_timeout 판정은 _last_seen_continuous 기준으로만 수행.
        # (재연결 성공이 반복돼도 elapsed_since_continuous는 계속 증가)
        if elapsed_since_continuous > self._miss_timeout:
            print(
                f"[Tracker] ✗ TARGET_LOST "
                f"(연속탐지 기준 {elapsed_since_continuous:.1f}s) "
                f"→ 사망 추정(가림/이탈 포함), 새 타겟 탐색"
            )
            self._target = None
            self._ghost  = None
            self._vx = self._vy = 0.0

            # min_conf 전달: TARGET_LOST 후 재탐색에도 동일 기준 적용
            new = select_by_confidence(monsters, min_conf=self._min_conf)
            if new:
                self._target               = new
                self._last_seen            = now
                self._last_seen_continuous = now
                print(f"[Tracker] ★ 새 타겟: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self.target, 0.0

        # ── miss 중: ghost 위치 반환, 공격 불가 ──────────────────────
        # self.target → _ghost (예측 위치 Detection) 반환
        # miss_elapsed = elapsed_since_continuous > 0.0 → main.py miss_elapsed==0.0 조건 미충족
        # → 공격 실행 없음. ghost는 오버레이 표시 전용.
        return self.target, elapsed_since_continuous

    def reset(self):
        self._target               = None
        self._ghost                = None
        self._vx = self._vy        = 0.0
        self._last_seen            = 0.0
        self._last_seen_continuous = 0.0
        self._prev_t               = 0.0
        print("[Tracker] 리셋")
