"""
target_selector.py
------------------
타겟 선택 + 추적 로직.

동작 원칙:
  1. 타겟 없음 → cy >= MIN_TARGET_CY 조건의 몬스터 중 confidence 최고값 선택
  2. 타겟 고정 후 → 속도 벡터 + ghost 예측 위치 기반 추적
     - 연속 탐지 시: 직전 프레임 간격(dt)으로 속도 갱신
     - miss 중: 속도 벡터로 ghost(예측 위치) 계산 → ref로 사용
     - 재연결 탐색: ghost(or 마지막 위치) 기준 max_allowed_dist 이내 가장 가까운 몬스터
     - max_allowed_dist = max(max_speed * elapsed, min_search_r)
       → 속도 미확보 상태에서도 min_search_r(60px) 보장
  3. miss_timeout 초 이상 탐지 안 되면 → 사망 추정 → 새 타겟 선택

변수 역할 분리:
  dt      : 이번 update() 호출과 직전 호출 사이의 실제 경과시간 (속도 계산에 사용)
  elapsed : _last_seen(마지막 실제 탐지) 이후 경과시간 (ghost 범위·miss_timeout 판정에 사용)

ghost 용도:
  - 오버레이 표시: miss 중 박스가 이동 방향으로 따라감
  - 재연결 탐색 ref: 예측 위치 주변에서 탐색
  - 공격 좌표에는 절대 사용하지 않음 (실제 탐지된 detection만 공격에 사용)

좌표 원칙:
  - 입력/출력 모두 ROI 기준 좌표 그대로. 변환 없음.
"""

import math
import time
import copy
from typing import List, Optional, Tuple
from detector import Detection

# 신규 타겟 선택 시 최소 cy 필터 (공격 판정의 min_cy와 동일)
# 이 값 미만인 몬스터는 타겟으로 선택하지 않음
MIN_TARGET_CY = 150

# 재연결 탐색 최소 보장 반경 (px)
# 속도 미확보 상태(초기 선택 직후) 또는 짧은 elapsed 구간에서
# max_speed * elapsed가 너무 작아지는 것을 방지
_MIN_SEARCH_R = 60


def select_by_confidence(detections: List[Detection]) -> Optional[Detection]:
    """
    cy >= MIN_TARGET_CY 조건을 만족하는 몬스터 중 confidence 가장 높은 것 반환.

    변경 전: cy 필터 없음 → cy=55, cy=110 같은 화면 상단 오탐도 타겟으로 선택됨
             → 공격 판정(cy >= min_cy)에서 막히지만 그 사이 실제 몬스터를 놓침
    변경 후: 타겟 선택 진입점에서 cy 필터 적용
             → 처음부터 공격 가능한 위치의 몬스터만 선택
    """
    monsters = [
        d for d in detections
        if d.class_id == 0 and d.cy >= MIN_TARGET_CY
    ]
    if not monsters:
        return None
    return max(monsters, key=lambda d: d.confidence)


def _make_ghost(src: Detection, cx: int, cy: int) -> Detection:
    """src 탐지 결과를 기반으로 cx,cy만 다른 예측 Detection 생성."""
    g = copy.copy(src)
    g.x = cx - src.w // 2
    g.y = cy - src.h // 2
    return g


class TargetTracker:
    """
    타겟 1개 고정 + 속도 벡터 기반 추적.

    핵심 개선:
      1. dt(프레임 간격)와 elapsed(마지막 탐지 후 경과) 역할 분리
         - 속도 계산: 연속 탐지 시 dt 사용, miss 후 재연결 시 elapsed 사용
         - ghost/miss_timeout: elapsed 사용 (변경 없음)

      2. max_allowed_dist = max(max_speed * elapsed, min_search_r)
         - 변경 전: max_speed * max(elapsed, 0.05) → 최소 12.5px (30fps 환경)
         - 변경 후: max(동적값, 60px) → 속도 미확보 상태에서도 60px 보장
         - 60px 근거: 로그상 재연결 성공 케이스 최소 이동량 46px + 여유

      3. vx=vy=0이어도 ghost 계산
         - 변경 전: vx != 0 or vy != 0 조건 → 초기 상태에서 ghost=None
         - 변경 후: 항상 ghost 계산 (vx=vy=0이면 ghost = 현재 위치 그대로)
         - ghost가 있으면 ref = ghost → 예측 위치 기준 탐색 활성화

      4. 속도 클램핑 방식 명확화
         - 연속 탐지: dt로 계산한 순간 속도를 max_speed로 클램핑
         - miss 후 재연결: elapsed로 계산한 평균 속도를 max_speed로 클램핑
    """

    def __init__(self,
                 miss_timeout: float = 3.5,
                 max_dist: float = 150,          # 하위 호환용 (미사용)
                 max_speed: float = 250,          # px/초
                 min_search_r: float = _MIN_SEARCH_R):  # 최소 탐색 반경 (px)
        self._target       = None
        self._ghost        = None   # miss 중 또는 직전 프레임 기반 예측 위치
        self._last_seen    = 0.0    # 마지막으로 실제 탐지된 time.time()
        self._miss_timeout = miss_timeout
        self._max_speed    = max_speed
        self._min_search_r = min_search_r

        # 속도 벡터 (px/초)
        self._vx = 0.0
        self._vy = 0.0

        # dt 계산용: 직전 update() 호출 시각
        self._prev_t = 0.0

    @property
    def target(self) -> Optional[Detection]:
        """현재 타겟 (miss 중이면 예측 ghost 반환, 오버레이용)."""
        return self._ghost if self._ghost is not None else self._target

    def update(self, detections: List[Detection]) -> Tuple[Optional[Detection], float]:
        """
        매 프레임 호출.

        반환: (실제 탐지 Detection or ghost, miss_elapsed)
          - miss_elapsed == 0.0: 이번 프레임에 실제 탐지 성공 → 공격 가능
          - miss_elapsed  > 0.0: miss 중 → 공격 불가 (main.py 조건 유지)

        ghost는 오버레이 표시 및 재연결 탐색 ref 용도로만 사용.
        반환되는 (target, 0.0) 쌍에서 target은 항상 실제 탐지된 Detection.
        """
        monsters = [d for d in detections if d.class_id == 0]
        now = time.time()

        # ── dt: 직전 프레임과의 실제 경과시간 (속도 계산용) ──────────
        # elapsed(마지막 탐지 후 총 경과)와 구별되는 별도 변수.
        # update()가 호출되지 않은 구간(프레임 드롭 등)도 반영.
        dt = now - self._prev_t if self._prev_t > 0 else 0.033
        self._prev_t = now

        # ── 타겟 없음 → 새 타겟 선택 ─────────────────────────────────
        if self._target is None:
            new = select_by_confidence(monsters)   # cy >= MIN_TARGET_CY 필터 포함
            if new:
                self._target    = new
                self._ghost     = None
                self._last_seen = now
                self._vx = self._vy = 0.0
                print(f"[Tracker] ★ 타겟 선택: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self.target, 0.0

        # ── elapsed: 마지막 실제 탐지 이후 경과시간 ──────────────────
        # ghost 예측 범위·miss_timeout 판정에 사용.
        elapsed = now - self._last_seen

        # ── ghost(예측 위치) 계산 ─────────────────────────────────────
        # 변경 전: vx != 0 or vy != 0 조건 → 초기 상태에서 ghost=None
        #           → ref가 마지막 위치로 고정 → max_allowed_dist만 적용
        # 변경 후: 항상 ghost 계산
        #           - vx=vy=0이면 ghost = _target 현재 위치 (예측 이동 없음)
        #           - vx/vy가 있으면 속도 방향으로 예측 위치 이동
        #           → 두 경우 모두 ghost가 존재 → ref = ghost 경로로 통일
        pred_cx = int(self._target.cx + self._vx * elapsed)
        pred_cy = int(self._target.cy + self._vy * elapsed)
        pred_cx = max(0, pred_cx)
        pred_cy = max(0, pred_cy)
        self._ghost = _make_ghost(self._target, pred_cx, pred_cy)

        # 탐색 기준점: 항상 ghost (위 계산으로 항상 존재)
        ref = self._ghost

        # ── 재연결 탐색 반경 ──────────────────────────────────────────
        # 변경 전: max_speed * max(elapsed, 0.05)
        #           → 30fps 첫 프레임: 250 * 0.05 = 12.5px (하한 클램핑)
        #           → 몬스터가 13px만 이동해도 재연결 실패
        # 변경 후: max(max_speed * elapsed, min_search_r)
        #           → 30fps 첫 프레임: max(250*0.033, 60) = max(8.25, 60) = 60px
        #           → 속도 미확보 상태에서도 60px 보장
        #           → elapsed가 누적될수록 동적 반경이 커지며 min_search_r을 추월
        max_allowed_dist = max(
            self._max_speed * min(elapsed, 1.0),  # 동적 반경 (상한: 250px)
            self._min_search_r                     # 최소 보장 반경 (60px)
        )

        # ── 가장 가까운 몬스터 탐색 (ref 기준) ───────────────────────
        best      = None
        best_dist = max_allowed_dist   # 이 거리 이내 + 최근접 조건

        for d in monsters:
            dist = math.hypot(d.cx - ref.cx, d.cy - ref.cy)
            if dist < best_dist:
                best_dist = dist
                best = d

        # ── 재연결 성공 ───────────────────────────────────────────────
        if best is not None:
            prev_cx, prev_cy = self._target.cx, self._target.cy
            moved = math.hypot(best.cx - prev_cx, best.cy - prev_cy)

            # 속도 계산: 연속 탐지(miss 없음) vs miss 후 재연결 분기
            # 연속 탐지: dt(실제 프레임 간격)로 계산 → 순간 속도, 정밀
            # miss 후 재연결: elapsed(누적 시간)로 계산 → 평균 속도, 노이즈 있음
            #
            # 판별 기준: elapsed가 dt의 2배 이상이면 miss가 있었던 것으로 판단.
            # dt와 elapsed가 유사하면(~1프레임) 연속 탐지로 처리.
            was_continuous = (elapsed < dt * 2.0)
            time_divisor   = dt if was_continuous else elapsed

            if time_divisor > 0:
                new_vx = (best.cx - prev_cx) / time_divisor
                new_vy = (best.cy - prev_cy) / time_divisor

                # 속도 클램핑 (방향 유지, 크기만 제한)
                speed = math.hypot(new_vx, new_vy)
                if speed > self._max_speed:
                    scale = self._max_speed / speed
                    new_vx *= scale
                    new_vy *= scale

                self._vx = new_vx
                self._vy = new_vy

            self._target    = best
            self._ghost     = None
            self._last_seen = now

            if moved > 20:
                mode_str = "연속" if was_continuous else f"miss후({elapsed:.2f}s)"
                print(f"[Tracker] 이동({mode_str}): ({prev_cx},{prev_cy}) → "
                      f"({best.cx},{best.cy})  Δ{moved:.0f}px  "
                      f"vx={self._vx:.0f} vy={self._vy:.0f}")

            # 실제 탐지된 Detection 반환 → miss_elapsed=0.0 → 공격 가능
            return self._target, 0.0

        # ── 탐지 실패 → miss 타이머 체크 ─────────────────────────────
        if elapsed > self._miss_timeout:
            print(f"[Tracker] ✗ 소실 ({elapsed:.1f}s) → 사망 추정, 새 타겟 탐색")
            self._target = None
            self._ghost  = None
            self._vx = self._vy = 0.0

            new = select_by_confidence(monsters)   # cy >= MIN_TARGET_CY 필터 포함
            if new:
                self._target    = new
                self._last_seen = now
                print(f"[Tracker] ★ 새 타겟: cx={new.cx} cy={new.cy} conf={new.confidence:.2f}")
            return self.target, 0.0

        # ── miss 중: ghost 위치로 오버레이 표시, 공격 불가 ───────────
        # self.target → _ghost (예측 위치) 반환
        # miss_elapsed = elapsed > 0.0 → main.py 공격 조건 미충족 → 공격 안 됨
        return self.target, elapsed

    def reset(self):
        self._target = None
        self._ghost  = None
        self._vx = self._vy = 0.0
        self._last_seen = 0.0
        self._prev_t    = 0.0
        print("[Tracker] 리셋")
