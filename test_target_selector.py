"""
test_target_selector.py
-----------------------
TargetTracker 및 select_by_confidence 단위 테스트.

구성:
  Section A: select_by_confidence (11건) — 이전 세션 기준 그대로 재현
  Section B: TargetTracker 기본 동작 (11건) — 이전 세션 기준 그대로 재현
  Section C: TARGET_LOST 기준점 분리 검증 (5건) — 이번 수정 검증용 신규
"""

import math
import time
import unittest
from unittest.mock import patch

# target_selector 를 직접 import.
# detector 의존성 때문에 Detection stub 을 먼저 만든다.
import sys
import types

# ── detector stub ──────────────────────────────────────────────────────────
# 실제 detector.py(YOLO 로드 필요)를 import하지 않고 최소 stub만 만든다.
detector_mod = types.ModuleType("detector")


class _Detection:
    """detector.Detection 의 최소 stub."""
    def __init__(self, class_id, cx, cy, confidence=0.90, w=60, h=80):
        self.class_id   = class_id
        self.cx         = cx
        self.cy         = cy
        self.confidence = confidence
        self.x          = cx - w // 2
        self.y          = cy - h // 2
        self.w          = w
        self.h          = h


detector_mod.Detection = _Detection
sys.modules["detector"] = detector_mod

# 이제 target_selector import 가능
from target_selector import (
    select_by_confidence,
    TargetTracker,
    MIN_TARGET_CY,
    _DEFAULT_MIN_CONF,
)


def det(cx, cy, conf=0.90, class_id=0):
    """테스트용 Detection 헬퍼."""
    return _Detection(class_id=class_id, cx=cx, cy=cy, confidence=conf)


# ══════════════════════════════════════════════════════════════════════════════
# Section A: select_by_confidence (11건)
# ══════════════════════════════════════════════════════════════════════════════
class TestSelectByConfidence(unittest.TestCase):

    # A01: 빈 리스트 → None
    def test_A01_empty(self):
        self.assertIsNone(select_by_confidence([]))

    # A02: class_id=1(아데나)만 있으면 → None
    def test_A02_no_monster(self):
        self.assertIsNone(select_by_confidence([det(400, 300, class_id=1)]))

    # A03: cy < MIN_TARGET_CY(150) → None
    def test_A03_cy_too_small(self):
        self.assertIsNone(select_by_confidence([det(400, MIN_TARGET_CY - 1)]))

    # A04: cy == MIN_TARGET_CY → 선택됨
    def test_A04_cy_exact_boundary(self):
        d = det(400, MIN_TARGET_CY)
        self.assertIs(select_by_confidence([d]), d)

    # A05: confidence < min_conf → None
    def test_A05_conf_below_min(self):
        self.assertIsNone(select_by_confidence([det(400, 300, conf=0.15)], min_conf=0.20))

    # A06: confidence == min_conf → 선택됨
    def test_A06_conf_exact_boundary(self):
        d = det(400, 300, conf=0.20)
        self.assertIs(select_by_confidence([d], min_conf=0.20), d)

    # A07: 여러 후보 중 confidence 최고값 선택
    def test_A07_max_confidence(self):
        low  = det(400, 300, conf=0.30)
        high = det(600, 400, conf=0.85)
        self.assertIs(select_by_confidence([low, high], min_conf=0.20), high)

    # A08: cy 필터와 conf 필터 동시 적용
    def test_A08_combined_filters(self):
        bad_cy   = det(400, 100, conf=0.90)   # cy 미달
        bad_conf = det(400, 300, conf=0.10)   # conf 미달
        good     = det(400, 300, conf=0.50)
        result   = select_by_confidence([bad_cy, bad_conf, good], min_conf=0.20)
        self.assertIs(result, good)

    # A09: 모두 통과하는 후보가 하나 → 그것 반환
    def test_A09_single_valid(self):
        d = det(500, 500, conf=0.75)
        self.assertIs(select_by_confidence([d]), d)

    # A10: default min_conf 값 확인 (0.20)
    def test_A10_default_min_conf(self):
        self.assertAlmostEqual(_DEFAULT_MIN_CONF, 0.20)

    # A11: 동일 confidence 두 후보 — 둘 다 반환 가능(max는 첫 번째 선택)
    def test_A11_tie_confidence(self):
        d1 = det(300, 300, conf=0.50)
        d2 = det(700, 400, conf=0.50)
        result = select_by_confidence([d1, d2], min_conf=0.20)
        self.assertIn(result, [d1, d2])


# ══════════════════════════════════════════════════════════════════════════════
# Section B: TargetTracker 기본 동작 (11건)
# ══════════════════════════════════════════════════════════════════════════════
class TestTargetTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = TargetTracker(miss_timeout=3.5, max_speed=250, min_conf=0.20)

    # B01: 초기 상태에서 탐지 없으면 (None, 0.0)
    def test_B01_initial_no_detection(self):
        tgt, elapsed = self.tracker.update([])
        self.assertIsNone(tgt)
        self.assertEqual(elapsed, 0.0)

    # B02: 타겟 선택 — conf 미달 → 선택 안 됨
    def test_B02_conf_below_min_not_selected(self):
        tgt, elapsed = self.tracker.update([det(400, 300, conf=0.15)])
        self.assertIsNone(tgt)

    # B03: 타겟 선택 — cy 미달 → 선택 안 됨
    def test_B03_cy_below_min_not_selected(self):
        tgt, elapsed = self.tracker.update([det(400, 100, conf=0.80)])
        self.assertIsNone(tgt)

    # B04: 정상 탐지 → miss_elapsed == 0.0 반환
    def test_B04_normal_detection_miss_elapsed_zero(self):
        d = det(400, 300, conf=0.80)
        tgt, elapsed = self.tracker.update([d])
        self.assertIsNotNone(tgt)
        self.assertEqual(elapsed, 0.0)

    # B05: 연속 탐지 2프레임 → 두 번째도 miss_elapsed == 0.0
    def test_B05_continuous_detection(self):
        with patch("time.time", side_effect=[100.0, 100.033]):
            self.tracker.update([det(400, 300, conf=0.80)])
            tgt, elapsed = self.tracker.update([det(410, 305, conf=0.80)])
        self.assertEqual(elapsed, 0.0)

    # B06: miss 1프레임 → miss_elapsed > 0.0 (ghost 반환)
    def test_B06_miss_one_frame(self):
        with patch("time.time", side_effect=[100.0, 100.033, 100.066]):
            self.tracker.update([det(400, 300, conf=0.80)])
            self.tracker.update([det(400, 300, conf=0.80)])
            tgt, elapsed = self.tracker.update([])  # miss
        self.assertGreater(elapsed, 0.0)

    # B07: miss_timeout 초과 → TARGET_LOST → None 반환
    def test_B07_target_lost_after_timeout(self):
        # t=0: 타겟 선택
        with patch("time.time", return_value=100.0):
            self.tracker.update([det(400, 300, conf=0.80)])
        # t=0.1: 연속탐지로 _last_seen_continuous 갱신
        with patch("time.time", return_value=100.1):
            self.tracker.update([det(400, 300, conf=0.80)])
        # t=103.7: 3.6s 경과 (miss_timeout=3.5s 초과)
        with patch("time.time", return_value=103.7):
            tgt, elapsed = self.tracker.update([])
        # TARGET_LOST → 새 타겟 없음 → None
        self.assertIsNone(tgt)
        self.assertEqual(elapsed, 0.0)

    # B08: TARGET_LOST 후 새 타겟 즉시 선택
    def test_B08_retarget_after_lost(self):
        with patch("time.time", return_value=100.0):
            self.tracker.update([det(400, 300, conf=0.80)])
        with patch("time.time", return_value=104.0):
            new_d = det(500, 400, conf=0.70)
            tgt, elapsed = self.tracker.update([new_d])
        self.assertIsNotNone(tgt)
        self.assertEqual(elapsed, 0.0)

    # B09: reset() 후 상태 초기화
    def test_B09_reset_clears_state(self):
        self.tracker.update([det(400, 300, conf=0.80)])
        self.tracker.reset()
        tgt, elapsed = self.tracker.update([])
        self.assertIsNone(tgt)
        self.assertEqual(elapsed, 0.0)

    # B10: 재연결 탐색 — ghost 반경 안에 몬스터 있으면 miss_elapsed=0.0 반환
    def test_B10_reconnect_within_radius(self):
        with patch("time.time", side_effect=[100.0, 100.033, 100.5]):
            self.tracker.update([det(400, 300, conf=0.80)])
            self.tracker.update([det(400, 300, conf=0.80)])
            # miss 후 0.467s — 반경 = max(250*0.467, 60) = ~116px
            # 같은 위치(거리=0)에 몬스터 재등장 → 재연결 성공
            tgt, elapsed = self.tracker.update([det(400, 300, conf=0.80)])
        self.assertEqual(elapsed, 0.0)

    # B11: conf 미달 탐지는 재연결에서 제외
    def test_B11_low_conf_not_reconnected(self):
        with patch("time.time", side_effect=[100.0, 100.033, 100.5]):
            self.tracker.update([det(400, 300, conf=0.80)])
            self.tracker.update([det(400, 300, conf=0.80)])
            # miss 후 conf=0.10 탐지 → 재연결 불가
            tgt, elapsed = self.tracker.update([det(400, 300, conf=0.10)])
        self.assertGreater(elapsed, 0.0)   # miss 상태 유지


# ══════════════════════════════════════════════════════════════════════════════
# Section C: TARGET_LOST 기준점 분리 검증 (5건) — 이번 수정 핵심 검증
# ══════════════════════════════════════════════════════════════════════════════
class TestTargetLostTimerIsolation(unittest.TestCase):
    """
    _last_seen_continuous 분리 수정 검증.

    핵심 불변식:
      miss 후 재연결(was_continuous=False)이 반복되어도
      _last_seen_continuous는 갱신되지 않는다.
      따라서 elapsed_since_continuous는 계속 증가하고,
      마지막 연속탐지 기준 miss_timeout 초 후에 TARGET_LOST가 발생한다.
    """

    TIMEOUT = 3.5   # miss_timeout

    def _make_tracker(self):
        return TargetTracker(
            miss_timeout=self.TIMEOUT,
            max_speed=250,
            min_search_r=200,   # 넓게 — 재연결 반경 충분히 확보
            min_conf=0.20,
        )

    # C01: 정상 연속탐지 중에는 timer 기준이 갱신된다
    #   시나리오: t=0 선택, t=0.1~1.0 연속탐지 10프레임
    #   검증: miss_timeout 이후에도 연속탐지 중에는 TARGET_LOST 미발생
    def test_C01_continuous_resets_timer(self):
        tracker = self._make_tracker()
        times = [float(i) * 0.1 for i in range(11)]  # 0.0~1.0, 11개
        call_n = 0

        def fake_time():
            nonlocal call_n
            t = times[call_n] if call_n < len(times) else times[-1]
            call_n += 1
            return t

        with patch("time.time", side_effect=fake_time):
            # t=0.0: 타겟 선택
            tracker.update([det(400, 300, conf=0.80)])
            last_elapsed = None
            for i in range(1, 11):
                # t=0.1~1.0: 연속탐지 (같은 위치 근방)
                tgt, e = tracker.update([det(400 + i, 300, conf=0.80)])
                last_elapsed = e

        # 마지막 연속탐지 직후 → miss_elapsed == 0.0
        self.assertEqual(last_elapsed, 0.0,
            "연속탐지 성공 직후 miss_elapsed 는 0.0 이어야 함")
        # _last_seen_continuous 가 갱신됐으므로 내부적으로 elapsed_since_continuous 작음
        self.assertAlmostEqual(
            tracker._last_seen_continuous, tracker._last_seen, delta=0.01,
            msg="_last_seen_continuous 가 연속탐지 최신 시각으로 갱신되어야 함"
        )

    # C02: 재연결 성공이 반복되더라도 miss_timeout 기준은 리셋되지 않는다
    #   시나리오:
    #     t=0.0  : 타겟 선택
    #     t=0.05 : 연속탐지 1회 (was_continuous=True) → _last_seen_continuous = 0.05
    #     t=0.5  : 재연결 1회 (was_continuous=False) → _last_seen_continuous 갱신 금지
    #     t=1.0  : 재연결 2회 (was_continuous=False) → _last_seen_continuous 갱신 금지
    #     t=2.0  : 재연결 3회 (was_continuous=False) → _last_seen_continuous 갱신 금지
    #     t=3.0  : 재연결 4회 (was_continuous=False) → _last_seen_continuous 갱신 금지
    #     t=3.7  : miss (탐지 없음) → elapsed_since_continuous = 3.7 - 0.05 = 3.65 > 3.5
    #   검증: t=3.7에서 TARGET_LOST 발생 (tgt=None)
    def test_C02_reconnect_does_not_reset_miss_timer(self):
        tracker = self._make_tracker()
        times = [0.0, 0.05, 0.5, 1.0, 2.0, 3.0, 3.7]
        call_n = 0

        def fake_time():
            nonlocal call_n
            t = times[min(call_n, len(times) - 1)]
            call_n += 1
            return t

        with patch("time.time", side_effect=fake_time):
            # t=0.0: 타겟 선택 → _last_seen_continuous = 0.0
            tracker.update([det(400, 300, conf=0.80)])

            # t=0.05: 연속탐지 → was_continuous=True → _last_seen_continuous = 0.05
            tracker.update([det(402, 300, conf=0.80)])

            # t=0.5~3.0: 재연결 4회 — 각각 miss 후 같은 몬스터가 반경 안에 있음
            # was_continuous=False (elapsed > dt*2) → _last_seen_continuous 갱신 금지
            tracker.update([det(420, 310, conf=0.80)])   # t=0.5
            tracker.update([det(450, 320, conf=0.80)])   # t=1.0
            tracker.update([det(500, 340, conf=0.80)])   # t=2.0
            tracker.update([det(550, 360, conf=0.80)])   # t=3.0

            # t=3.7: miss → elapsed_since_continuous = 3.7 - 0.05 = 3.65 > 3.5
            tgt, elapsed = tracker.update([])

        # TARGET_LOST 발생: 새 타겟 없음 → None
        self.assertIsNone(tgt,
            "재연결 4회 후 miss → elapsed_since_continuous > 3.5 → TARGET_LOST 발생해야 함")
        self.assertEqual(elapsed, 0.0)

    # C03: 재연결 후 miss 중에 miss_elapsed 가 elapsed_since_continuous 기준으로 반환된다
    #   시나리오:
    #     t=1000.0 : 타겟 선택 → _last_seen_continuous = 1000.0
    #     t=1000.03: 연속탐지 (was_continuous=True) → _last_seen_continuous = 1000.03
    #     t=1001.0 : 재연결 (was_continuous=False) → _last_seen_continuous = 1000.03 유지
    #     t=1002.0 : miss → elapsed_since_continuous = 1002.0 - 1000.03 = 1.97
    #
    #   참고: _prev_t > 0 조건 때문에 t=0.0 시작 시 두 번째 호출에서 dt=0.033(기본값)
    #         → elapsed=0.1 > dt*2=0.066 → was_continuous=False
    #         현실적인 Unix 타임스탬프(1000.0+)를 사용하면 이 문제 없음.
    def test_C03_miss_elapsed_reflects_continuous_base(self):
        tracker = self._make_tracker()
        # t=1000.0: 선택, t=1000.03: 연속(dt=0.03 → was_continuous=(0.03<0.06)=True)
        # t=1001.0: 재연결(elapsed=0.97, dt=0.03 → was_continuous=False)
        # t=1002.0: miss
        T0 = 1000.0
        times = [T0, T0 + 0.03, T0 + 1.0, T0 + 2.0]
        call_n = 0

        def fake_time():
            nonlocal call_n
            t = times[min(call_n, len(times) - 1)]
            call_n += 1
            return t

        with patch("time.time", side_effect=fake_time):
            tracker.update([det(400, 300, conf=0.80)])         # t=1000.00: 선택
            tracker.update([det(402, 300, conf=0.80)])         # t=1000.03: 연속탐지
            tracker.update([det(450, 320, conf=0.80)])         # t=1001.00: 재연결
            tgt, elapsed = tracker.update([])                  # t=1002.00: miss

        # _last_seen_continuous = 1000.03 (연속탐지 시 갱신)
        # elapsed_since_continuous = 1002.0 - 1000.03 = 1.97
        expected = (T0 + 2.0) - (T0 + 0.03)   # = 1.97
        self.assertAlmostEqual(elapsed, expected, delta=0.02,
            msg="miss_elapsed 는 _last_seen_continuous(마지막 연속탐지) 기준이어야 함")

    # C04: 마지막 연속탐지로부터 정확히 miss_timeout 초 후 TARGET_LOST 발생
    #   시나리오 (실제 로그 시나리오 재현):
    #     t=11.715 : 마지막 연속탐지 → _last_seen_continuous = 11.715
    #     t=12.0~19.0 : 재연결 6회 (was_continuous=False)
    #     t=15.215 = 11.715 + 3.5 : TARGET_LOST 발생 예상 시각
    #   검증: 연속탐지 기준 3.5s 초과 직후 miss에서 TARGET_LOST
    def test_C04_target_lost_at_continuous_plus_timeout(self):
        tracker = self._make_tracker()
        # times: 선택, 연속탐지, 재연결*6, TARGET_LOST 체크
        t_continuous = 11.715
        times = (
            [11.0, t_continuous]                          # 선택, 마지막 연속탐지
            + [12.2, 13.4, 14.5, 15.6, 16.8, 17.9]       # 재연결 6회
            + [15.3]                                       # t=15.3 > 11.715+3.5=15.215 → LOST
        )
        call_n = 0

        def fake_time():
            nonlocal call_n
            t = times[min(call_n, len(times) - 1)]
            call_n += 1
            return t

        with patch("time.time", side_effect=fake_time):
            tracker.update([det(400, 300, conf=0.80)])         # t=11.0: 선택
            tracker.update([det(402, 300, conf=0.80)])         # t=11.715: 마지막 연속탐지
            # 재연결 6회
            for _ in range(6):
                tracker.update([det(500, 350, conf=0.80)])
            # t=15.3: miss → elapsed_since_continuous = 15.3 - 11.715 = 3.585 > 3.5
            tgt, elapsed = tracker.update([])

        self.assertIsNone(tgt,
            "마지막 연속탐지 기준 3.5s 초과 → TARGET_LOST 발생해야 함")

    # C05: TARGET_LOST 후 재탐색 동작이 정상 동작한다
    #   시나리오:
    #     t=1000.0  : 타겟 선택 → _last_seen_continuous = 1000.0
    #     t=1000.03 : 연속탐지 → _last_seen_continuous = 1000.03
    #     t=1004.0  : miss → elapsed_since_continuous = 3.97 > 3.5 → TARGET_LOST
    #                 (참고: min_search_r=200, max_allowed_dist=max(250*3.97, 200)=992px
    #                  new_target(600,400) vs ghost 위치 — 거리가 반경보다 작으면 재연결
    #                  TARGET_LOST 가 재연결보다 먼저 일어나지는 않음. 재연결 후 LOST 없음.
    #                  → 새 타겟을 반경 밖에 배치하거나 탐지 목록에서 제외해야 함)
    #
    #   구현 전략: LOST 프레임과 재선택 프레임을 분리
    #     - t=1004.0: miss (탐지 없음) → TARGET_LOST 확인
    #     - t=1004.1: 새 타겟 존재 프레임 → 새 선택 확인
    def test_C05_retarget_after_lost_resets_continuous_timer(self):
        tracker = self._make_tracker()
        T0 = 1000.0
        times = [T0, T0 + 0.03, T0 + 4.0, T0 + 4.1]
        call_n = 0

        def fake_time():
            nonlocal call_n
            t = times[min(call_n, len(times) - 1)]
            call_n += 1
            return t

        new_target = det(600, 400, conf=0.75)

        with patch("time.time", side_effect=fake_time):
            tracker.update([det(400, 300, conf=0.80)])         # t=T0+0.00: 선택
            tracker.update([det(402, 300, conf=0.80)])         # t=T0+0.03: 연속
            tgt_lost, e_lost = tracker.update([])              # t=T0+4.00: LOST (탐지 없음)
            tgt_new, e_new   = tracker.update([new_target])   # t=T0+4.10: 새 타겟

        # t=T0+4.0: TARGET_LOST → 새 타겟 없음 → None
        self.assertIsNone(tgt_lost,
            "TARGET_LOST 시 탐지 없음 → None 이어야 함")
        self.assertEqual(e_lost, 0.0)

        # t=T0+4.1: 새 타겟 즉시 선택
        self.assertIsNotNone(tgt_new,
            "TARGET_LOST 직후 다음 프레임에 새 타겟이 있으면 즉시 선택되어야 함")
        self.assertEqual(e_new, 0.0)

        # _last_seen_continuous 도 새 타겟 선택 시각으로 갱신됐는지 확인
        self.assertAlmostEqual(
            tracker._last_seen_continuous, T0 + 4.1, delta=0.01,
            msg="새 타겟 선택 후 _last_seen_continuous 가 현재 시각으로 갱신되어야 함"
        )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
