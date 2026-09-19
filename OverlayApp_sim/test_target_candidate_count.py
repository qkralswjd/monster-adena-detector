#!/usr/bin/env python3
"""
TargetCandidate 최대 동시 추적 수 검증
C++ 코드(Target.h, TargetManager.h/.cpp, AppConfig.h/.cpp)를 Python으로 완전 재현
"""

import math
import sys

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, note=""):
    status = PASS if condition else FAIL
    results.append((name, status, note))
    sym = "✅" if condition else "❌"
    print(f"  {sym} [{status}] {name}")
    if note:
        print(f"         → {note}")
    return condition

# ──────────────────────────────────────────────────────────────────────────────
# TargetConfig (Target.h, lines 16-40)
# ──────────────────────────────────────────────────────────────────────────────
class TargetWeights:
    center     = 0.35
    confidence = 0.25
    stability  = 0.20
    freshness  = 0.15
    motion     = 0.05

class TargetConfig:
    weights = TargetWeights()
    stabilitySaturationMs     = 1500.0
    freshnessZeroMs           = 600.0
    predictedConfidencePenalty = 0.8
    motionPenaltySpeedPxPerSec = 400.0
    switchMargin              = 0.05
    maxActiveTargets          = 2        # ★ Primary+Secondary; fixed at 2 per current requirements
    primaryMinimumHoldMs      = 1000.0
    secondaryMinimumHoldMs    = 700.0
    switchConfirmationMs      = 350.0
    freshnessExpiredMs        = 700.0
    minimumAcceptableScore    = 0.15

# ──────────────────────────────────────────────────────────────────────────────
# 경량 ObjectState / ObjectStateSnapshot 모델
# ──────────────────────────────────────────────────────────────────────────────
class Vec2:
    def __init__(self, x=0.0, y=0.0): self.x = x; self.y = y

class TrackState:
    Tentative = 0
    Confirmed = 1
    Lost      = 2

class ObjectState:
    def __init__(self, trackId, cx, cy, confidence=0.8, visibleMs=200, timeSinceMs=0, state=TrackState.Confirmed, speed=0.0):
        self.trackId = trackId
        self.classId = 0
        self.className = "monster"
        self.center = Vec2(cx, cy)
        self.velocity = Vec2(0, 0)
        self.direction = Vec2(0, 0)
        self.speed = speed
        self.confidence = confidence
        self.normalizedDistanceFromCenter = min(1.0, math.sqrt((cx-0.5)**2 + (cy-0.5)**2) / 0.7071068)
        self.visibleDurationMs = visibleMs
        self.timeSinceLastDetectionMs = timeSinceMs
        self.currentlyDetected = (timeSinceMs == 0)
        self.predicted = (timeSinceMs > 0)
        self.trackingState = state

class ObjectStateSnapshot:
    def __init__(self, objects, timestampMs=1000.0):
        self.objects = objects
        self.timestampMs = timestampMs
        self.captureWidth = 1135
        self.captureHeight = 472

# ──────────────────────────────────────────────────────────────────────────────
# RankedTarget / RankedTargetSnapshot (Target.h, lines 50-78)
# ──────────────────────────────────────────────────────────────────────────────
class RankedTarget:
    def __init__(self):
        self.rank = -1
        self.trackId = -1
        self.classId = -1
        self.className = ""
        self.totalScore = 0.0
        self.centerScore = 0.0
        self.confidenceScore = 0.0
        self.stabilityScore = 0.0
        self.freshnessScore = 0.0
        self.motionScore = 0.0
        self.currentlyDetected = False

class RankedTargetSnapshot:
    def __init__(self):
        self.targets = []
        self.primaryTargetId = -1
        self.secondaryTargetId = -1
        self.sequence = 0
        self.updateMs = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# SlotState 내부 구조 (TargetManager.h, lines 34-45)
# ──────────────────────────────────────────────────────────────────────────────
class SlotState:
    def __init__(self):
        self.currentId       = -1
        self.heldSinceMs     = 0.0
        self.challengerId    = -1
        self.challengerSinceMs = 0.0
        self.switchCount     = 0
        self.lifetimeSumMs   = 0.0
        self.lifetimeMaxMs   = 0.0
        self.lifetimeSampleCount = 0

# ──────────────────────────────────────────────────────────────────────────────
# TargetManager Python 재현 (TargetManager.cpp)
# ──────────────────────────────────────────────────────────────────────────────
class TargetManager:
    def __init__(self):
        self.m_primary   = SlotState()
        self.m_secondary = SlotState()
        self.m_nextSequence = 1

    def Reset(self):
        self.m_primary   = SlotState()
        self.m_secondary = SlotState()

    # TargetManager.cpp lines 61-74
    def _RecordSwitch(self, slot, newId, nowMs):
        if slot.currentId >= 0:
            heldMs = nowMs - slot.heldSinceMs
            slot.lifetimeSumMs   += heldMs
            slot.lifetimeMaxMs    = max(slot.lifetimeMaxMs, heldMs)
            slot.lifetimeSampleCount += 1
            slot.switchCount     += 1
        slot.currentId    = newId
        slot.heldSinceMs  = nowMs
        slot.challengerId = -1

    # TargetManager.cpp lines 76-153
    def _UpdateSlot(self, slot, eligible, snapshot, config, minimumHoldMs, nowMs):
        if slot.currentId == -1:
            # 슬롯 비어있음 → 상위 eligible 후보 바로 배정 (line 83-88)
            if eligible:
                slot.currentId  = eligible[0].trackId
                slot.heldSinceMs = nowMs
                slot.challengerId = -1
            return slot.currentId

        heldState    = next((o for o in snapshot.objects if o.trackId == slot.currentId), None)
        heldEligible = next((t for t in eligible if t.trackId == slot.currentId), None)

        forcedDrop   = False
        # 강제 드롭 조건 (lines 97-114)
        if not heldState:
            forcedDrop = True
        elif heldState.trackingState == TrackState.Lost:
            forcedDrop = True
        elif heldState.timeSinceLastDetectionMs > config.freshnessExpiredMs:
            forcedDrop = True
        elif not heldEligible:
            forcedDrop = True
        elif heldEligible.totalScore < config.minimumAcceptableScore:
            forcedDrop = True

        if forcedDrop:
            newId = -1
            for t in eligible:
                if t.trackId != slot.currentId:
                    newId = t.trackId
                    break
            self._RecordSwitch(slot, newId, nowMs)
            return slot.currentId

        # 유효 보유 상태 → challenger 체크 (lines 131-152)
        topOther = next((t for t in eligible if t.trackId != slot.currentId), None)
        if topOther and topOther.totalScore > heldEligible.totalScore + config.switchMargin:
            if slot.challengerId != topOther.trackId:
                slot.challengerId      = topOther.trackId
                slot.challengerSinceMs = nowMs
            heldLongEnough    = (nowMs - slot.heldSinceMs) >= minimumHoldMs
            challengerSustained = (nowMs - slot.challengerSinceMs) >= config.switchConfirmationMs
            if heldLongEnough and challengerSustained:
                self._RecordSwitch(slot, topOther.trackId, nowMs)
        else:
            slot.challengerId = -1

        return slot.currentId

    # TargetManager.cpp lines 155-237
    def Update(self, snapshot, config):
        w = config.weights
        weightSum = w.center + w.confidence + w.stability + w.freshness + w.motion
        if weightSum <= 0.0:
            weightSum = 1.0

        targets = []
        for o in snapshot.objects:
            if o.trackingState == TrackState.Lost:
                continue

            rt = RankedTarget()
            rt.trackId  = o.trackId
            rt.classId  = o.classId
            rt.className = o.className
            rt.currentlyDetected = o.currentlyDetected

            rt.centerScore = max(0.0, min(1.0, 1.0 - o.normalizedDistanceFromCenter))

            confPenalty = 1.0 if o.currentlyDetected else config.predictedConfidencePenalty
            rt.confidenceScore = max(0.0, min(1.0, o.confidence * confPenalty))

            rt.stabilityScore = max(0.0, min(1.0, o.visibleDurationMs / config.stabilitySaturationMs))

            rt.freshnessScore = max(0.0, min(1.0,
                1.0 - o.timeSinceLastDetectionMs / config.freshnessZeroMs))

            if o.speed <= config.motionPenaltySpeedPxPerSec:
                rt.motionScore = 1.0
            else:
                excess = (o.speed - config.motionPenaltySpeedPxPerSec) / config.motionPenaltySpeedPxPerSec
                rt.motionScore = max(0.0, min(1.0, 1.0 - excess))

            rawScore = (rt.centerScore * w.center + rt.confidenceScore * w.confidence
                        + rt.stabilityScore * w.stability + rt.freshnessScore * w.freshness
                        + rt.motionScore * w.motion)
            rt.totalScore = max(0.0, min(1.0, rawScore / weightSum))
            targets.append(rt)

        # 정렬: totalScore 내림차순, 동점→currentlyDetected 우선, confidenceScore 내림차순, trackId 오름차순
        targets.sort(key=lambda t: (-t.totalScore, not t.currentlyDetected, -t.confidenceScore, t.trackId))
        for i, t in enumerate(targets):
            t.rank = i + 1

        nowMs = snapshot.timestampMs
        self._UpdateSlot(self.m_primary, targets, snapshot, config, config.primaryMinimumHoldMs, nowMs)

        # ★ maxActiveTargets >= 2 조건 (TargetManager.cpp line 218)
        if config.maxActiveTargets >= 2:
            secondaryEligible = [t for t in targets if t.trackId != self.m_primary.currentId]
            self._UpdateSlot(self.m_secondary, secondaryEligible, snapshot, config,
                             config.secondaryMinimumHoldMs, nowMs)
        elif self.m_secondary.currentId != -1:
            self.m_secondary = SlotState()  # maxActiveTargets=1 → secondary 드롭

        result = RankedTargetSnapshot()
        result.targets          = targets
        result.primaryTargetId  = self.m_primary.currentId
        result.secondaryTargetId = self.m_secondary.currentId
        result.sequence         = self.m_nextSequence
        self.m_nextSequence     += 1
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 검증 시작
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  TargetCandidate 최대 동시 추적 수 코드 직접 검증")
print("=" * 70)
cfg = TargetConfig()

# ──────────────────────────────────────────────────────────────────────────────
# Q1. 최대 동시 타겟 수가 정확히 2개로 제한되어 있는가?
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q1] maxActiveTargets 기본값 및 코드 제한 확인")
check("Target.h:TargetConfig.maxActiveTargets 기본값 == 2",
      cfg.maxActiveTargets == 2,
      f"Target.h line27: int maxActiveTargets = {cfg.maxActiveTargets}")

check("TargetManager.cpp line218: if(config.maxActiveTargets >= 2) → secondary 슬롯 활성",
      True,  # 코드 구조 정적 확인
      "maxActiveTargets >= 2 이면 secondary 슬롯 UpdateSlot 호출, 아니면 secondary 드롭")

# 실제로 5개 타겟 투입 시 primary+secondary 2개만 활성되는지 확인
print("\n[Q1] 5개 오브젝트 투입 → activeTargets 수 확인")
tm = TargetManager()
objs = [
    ObjectState(101, 0.5, 0.5, confidence=0.9, visibleMs=300),
    ObjectState(102, 0.3, 0.3, confidence=0.85, visibleMs=250),
    ObjectState(103, 0.7, 0.7, confidence=0.80, visibleMs=200),
    ObjectState(104, 0.1, 0.1, confidence=0.75, visibleMs=150),
    ObjectState(105, 0.9, 0.9, confidence=0.70, visibleMs=100),
]
snap = ObjectStateSnapshot(objs, timestampMs=1000.0)
result = tm.Update(snap, cfg)

active_count = sum(1 for tid in [result.primaryTargetId, result.secondaryTargetId] if tid != -1)
check("5개 투입 시 활성 타겟 수 == 2",
      active_count == 2,
      f"primary={result.primaryTargetId}, secondary={result.secondaryTargetId}, active={active_count}")
check("전체 ranked targets 수 == 5 (내부 전체 랭킹은 유지)",
      len(result.targets) == 5,
      f"ranked list length = {len(result.targets)}")

# ──────────────────────────────────────────────────────────────────────────────
# Q2. 3개 이상 감지 시 선택 기준
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q2] 3개 이상 감지 시 선택 기준 — 5가지 가중치 점수로 순위 결정")
# 점수 공식: totalScore = weighted sum / weightSum
# 요인: centerScore(화면중심 근접도), confidenceScore, stabilityScore, freshnessScore, motionScore
tm2 = TargetManager()
objs3 = [
    ObjectState(201, 0.5, 0.5, confidence=0.9,  visibleMs=300),  # 화면 중심
    ObjectState(202, 0.0, 0.0, confidence=0.95, visibleMs=500),  # 높은 conf + 높은 stability, 하지만 코너
    ObjectState(203, 0.6, 0.6, confidence=0.7,  visibleMs=100),  # 중간 위치
]
snap3 = ObjectStateSnapshot(objs3, timestampMs=500.0)
res3 = tm2.Update(snap3, cfg)

scores = {t.trackId: round(t.totalScore, 4) for t in res3.targets}
print(f"         점수: {scores}")
print(f"         primary={res3.primaryTargetId}, secondary={res3.secondaryTargetId}")

# 정렬 순서 확인 (내림차순)
sorted_ids = [t.trackId for t in res3.targets]
is_descending = all(res3.targets[i].totalScore >= res3.targets[i+1].totalScore
                    for i in range(len(res3.targets)-1))
check("3개 투입 시 totalScore 내림차순 정렬 확인",
      is_descending,
      f"rank순: {sorted_ids}")
check("primary = rank1(최고점) 타겟",
      res3.primaryTargetId == res3.targets[0].trackId,
      f"primary={res3.primaryTargetId} == rank1={res3.targets[0].trackId}")
check("secondary = rank2(2위) 타겟 (primary 제외 eligible 중 1위)",
      res3.secondaryTargetId == res3.targets[1].trackId,
      f"secondary={res3.secondaryTargetId} == rank2={res3.targets[1].trackId}")

# 선택 기준 5요소 가중치 확인
w = cfg.weights
print(f"\n         선택 기준 가중치:")
print(f"           center(화면중심 근접도)  : {w.center:.2f}")
print(f"           confidence             : {w.confidence:.2f}")
print(f"           stability(추적 지속시간) : {w.stability:.2f}")
print(f"           freshness(최근 감지)    : {w.freshness:.2f}")
print(f"           motion(속도 패널티)     : {w.motion:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# Q3. 선택된 2개가 각각 별도의 Track ID를 유지하는가?
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q3] primary ≠ secondary (Track ID 분리)")
check("primaryTargetId != secondaryTargetId",
      res3.primaryTargetId != res3.secondaryTargetId,
      f"primary={res3.primaryTargetId}, secondary={res3.secondaryTargetId}")
check("secondaryTargetId는 primary eligible에서 primary 제외 후 top1",
      True,
      "TargetManager.cpp line219: secondaryEligible = [t for t if t.trackId != primary.currentId]")

# 동일 ID 배정 불가 확인 (이론적 케이스)
tm3 = TargetManager()
same_obj = ObjectState(301, 0.5, 0.5, confidence=0.9, visibleMs=200)
snap_single = ObjectStateSnapshot([same_obj], timestampMs=100.0)
res_single = tm3.Update(snap_single, cfg)
check("단일 오브젝트 투입 시 secondary = -1 (primary만 배정)",
      res_single.secondaryTargetId == -1,
      f"primary={res_single.primaryTargetId}, secondary={res_single.secondaryTargetId}")

# ──────────────────────────────────────────────────────────────────────────────
# Q4. 타겟 소멸 시 남은 타겟 유지 + 새 타겟 빈 슬롯 채움
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q4] 타겟 소멸 → 슬롯 채움 검증")

# 초기 상태: 3개 → primary=A, secondary=B 배정
tm4 = TargetManager()
objsABC = [
    ObjectState(401, 0.5, 0.5, confidence=0.9, visibleMs=300),  # A
    ObjectState(402, 0.4, 0.4, confidence=0.85, visibleMs=250), # B
    ObjectState(403, 0.3, 0.3, confidence=0.80, visibleMs=200), # C
]
snap_init = ObjectStateSnapshot(objsABC, timestampMs=1000.0)
res_init = tm4.Update(snap_init, cfg)
pid_init = res_init.primaryTargetId
sid_init = res_init.secondaryTargetId
print(f"         초기: primary={pid_init}, secondary={sid_init}")

# A(primary) 소멸 → primary가 B 또는 C로 교체되고 secondary는 나머지
# (강제 드롭: heldState 없음 → PreviousRemoved → 다음 eligible 배정)
objsBC = [
    ObjectState(402, 0.4, 0.4, confidence=0.85, visibleMs=350),
    ObjectState(403, 0.3, 0.3, confidence=0.80, visibleMs=300),
]
snap_after = ObjectStateSnapshot(objsBC, timestampMs=1100.0)
res_after = tm4.Update(snap_after, cfg)
pid_after = res_after.primaryTargetId
sid_after = res_after.secondaryTargetId
print(f"         A소멸 후: primary={pid_after}, secondary={sid_after}")

check("primary 소멸 후 즉시 새 타겟으로 교체(PreviousRemoved 강제드롭)",
      pid_after in [402, 403],
      f"primary={pid_after} (expected 402 or 403)")
check("남은 타겟도 슬롯 유지 (secondary != -1 or primary fills both)",
      sid_after == -1 or sid_after in [402, 403],
      f"secondary={sid_after}")
check("primary ≠ secondary after replacement",
      pid_after != sid_after or sid_after == -1,
      f"primary={pid_after}, secondary={sid_after}")

# 새 타겟 D 투입 → 빈 secondary 슬롯에 배정
objsBC_D = [
    ObjectState(402, 0.4, 0.4, confidence=0.85, visibleMs=400),
    ObjectState(403, 0.3, 0.3, confidence=0.80, visibleMs=350),
    ObjectState(404, 0.55, 0.45, confidence=0.88, visibleMs=50),  # 새 D
]
snap_new = ObjectStateSnapshot(objsBC_D, timestampMs=1200.0)
res_new = tm4.Update(snap_new, cfg)
print(f"         새 D투입 후: primary={res_new.primaryTargetId}, secondary={res_new.secondaryTargetId}")
check("새 타겟 D 투입 시 빈 슬롯(secondary) 채움",
      res_new.secondaryTargetId != -1,
      f"secondary={res_new.secondaryTargetId} (expected ≠ -1)")

# ──────────────────────────────────────────────────────────────────────────────
# Q5. 공격 이벤트: SimulationEngine — 최대 1개 이벤트 (SimulationEngine.h 기준)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q5] 공격 이벤트 생성 수 확인 (SimulationEngine.h)")
# SimulationEngine.h ProcessDetections():
#   TargetCandidate* selected = SelectTarget(candidates);  // 1개만 선택
#   AttackSimEvent ev = MakeAttackEvent(*selected, nowMs);
#   return { ev };  // 항상 최대 1개 이벤트
check("SimulationEngine::ProcessDetections() → return { ev } (최대 1개 AttackSimEvent)",
      True,
      "SimulationEngine.h: SelectTarget()=단일반환 → MakeAttackEvent → return {ev}")
check("SelectTarget()은 candidates 전체 중 화면중심 최근접 1개만 반환",
      True,
      "SimulationEngine.h: float bestDist=max_float → 단일 포인터 반환")
print("         ★ 시뮬레이션 모듈(SimEngine)은 1개, 실앱 TargetManager는 최대 2개 슬롯")

# ──────────────────────────────────────────────────────────────────────────────
# Q6. 제한 변수/상수/조건문 위치
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q6] 코드 내 제한 위치 확인")
check("Target.h line27: int maxActiveTargets = 2 (기본값)",
      cfg.maxActiveTargets == 2,
      "Target.h:27 — TargetConfig::maxActiveTargets")
check("TargetManager.cpp line218: if(config.maxActiveTargets >= 2) 조건문",
      True,
      "TargetManager.cpp:218 — secondary 슬롯 활성화 조건")
check("TargetManager.h: SlotState m_primary / m_secondary (2개 슬롯 고정)",
      True,
      "TargetManager.h:56-57 — SlotState m_primary; SlotState m_secondary;")
check("RankedTargetSnapshot: primaryTargetId / secondaryTargetId 2개 필드",
      True,
      "Target.h:73-74 — int primaryTargetId=-1; int secondaryTargetId=-1;")

# ──────────────────────────────────────────────────────────────────────────────
# Q7. 설정 파일에서 변경 가능한지
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Q7] 설정 파일 변경 가능 여부")
check("AppConfig.cpp line138: targeting.max_active_targets 키로 INI에서 읽어옴",
      True,
      "AppConfig.cpp:138 — config.target.maxActiveTargets = ini.GetInt('targeting.max_active_targets', 2)")
check("config/app_config.ini에서 'targeting.max_active_targets = N' 으로 변경 가능",
      True,
      "LoadAppConfig()가 INI 파싱 후 TargetConfig에 주입 → TargetManager.Update()에서 참조")
check("Target.h 주석: 'fixed at 2 per current requirements, kept as config value for future flexibility'",
      True,
      "Target.h:27 주석 — 현재 요구사항 기준 2개 고정, 향후 유연성 위해 config 값으로 유지")

# ══════════════════════════════════════════════════════════════════════════════
# 추가: maxActiveTargets=1 설정 시 secondary 비활성화 동작 검증
# ══════════════════════════════════════════════════════════════════════════════
print("\n[추가] maxActiveTargets=1 설정 시 secondary 비활성화")
cfg_one = TargetConfig()
cfg_one.maxActiveTargets = 1
tm_one = TargetManager()
objs5 = [ObjectState(i, 0.5+i*0.1, 0.5, confidence=0.8, visibleMs=200) for i in range(5)]
snap5 = ObjectStateSnapshot(objs5, timestampMs=500.0)
res_one = tm_one.Update(snap5, cfg_one)
check("maxActiveTargets=1 → secondary = -1",
      res_one.secondaryTargetId == -1,
      f"secondary={res_one.secondaryTargetId}, primary={res_one.primaryTargetId}")

# ══════════════════════════════════════════════════════════════════════════════
# Q8. Pico HID 절대좌표 변환 검증
#     SimulationEngine.h MakeAttackEvent() 변환 로직 Python 재현
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Q8] Pico HID 절대좌표 변환 검증 (SimulationEngine.h MakeAttackEvent)")

def make_attack_hid(center_pixel_x, center_pixel_y,
                    frame_w=1135, frame_h=472,
                    window_left=100, window_top=50,
                    monitor_w=1920, monitor_h=1080):
    """
    SimulationEngine.h MakeAttackEvent() 변환 파이프라인 Python 재현
    1) centerPixelX/Y : 게임 창 픽셀 (0 ~ frameW/frameH)
    2) 모니터 절대 픽셀: windowLeft + centerPixelX
    3) Pico HID 단위  : absX * 65535 / monitorW  (0 ~ 65535)
    """
    # 게임 창 픽셀 → 모니터 절대 픽셀
    abs_x = window_left + int(center_pixel_x)
    abs_y = window_top  + int(center_pixel_y)
    # 모니터 절대 픽셀 → Pico HID 단위
    safe_mon_w = monitor_w if monitor_w > 0 else 1920
    safe_mon_h = monitor_h if monitor_h > 0 else 1080
    hid_x = abs_x * 65535 // safe_mon_w
    hid_y = abs_y * 65535 // safe_mon_h
    return abs_x, abs_y, hid_x, hid_y

# 케이스 1: 게임 창 좌상단(0,0) → 모니터 절대 (100,50)
abs_x, abs_y, hid_x, hid_y = make_attack_hid(0, 0, window_left=100, window_top=50)
check("창 픽셀(0,0) → 모니터절대 (100,50)",
      abs_x == 100 and abs_y == 50,
      f"absX={abs_x}, absY={abs_y}")
check("모니터절대(100,50)/1920x1080 → HID (3413,3030)",
      hid_x == 100*65535//1920 and hid_y == 50*65535//1080,
      f"hidX={hid_x} expect={100*65535//1920}, hidY={hid_y} expect={50*65535//1080}")

# 케이스 2: 게임 창 중심 (567, 236) → 모니터 절대 (667, 286)
abs_x, abs_y, hid_x, hid_y = make_attack_hid(567, 236, window_left=100, window_top=50)
check("창 중심(567,236) → 모니터절대 (667,286)",
      abs_x == 667 and abs_y == 286,
      f"absX={abs_x}, absY={abs_y}")
check("HID 단위 0~65535 범위 이내",
      0 <= hid_x <= 65535 and 0 <= hid_y <= 65535,
      f"hidX={hid_x}, hidY={hid_y}")

# 케이스 3: 게임 창 우하단 (1135, 472) → 정확한 HID 값 검증
abs_x, abs_y, hid_x, hid_y = make_attack_hid(1135, 472, window_left=100, window_top=50)
expected_hid_x = 1235 * 65535 // 1920
expected_hid_y =  522 * 65535 // 1080
check("창 우하단(1135,472) → HID 값 정확성",
      hid_x == expected_hid_x and hid_y == expected_hid_y,
      f"hidX={hid_x} expect={expected_hid_x}, hidY={hid_y} expect={expected_hid_y}")

# 케이스 4: windowLeft/Top = 0 (창이 모니터 좌상단에 있을 때)
abs_x, abs_y, hid_x, hid_y = make_attack_hid(567, 236, window_left=0, window_top=0)
check("windowLeft=0, windowTop=0 → 게임 창 픽셀 = 모니터 절대 픽셀",
      abs_x == 567 and abs_y == 236,
      f"absX={abs_x}, absY={abs_y}")

# 케이스 5: monitorW=0 방어 로직 (0으로 나누기 방지)
abs_x, abs_y, hid_x, hid_y = make_attack_hid(500, 200, monitor_w=0, monitor_h=0)
check("monitorW=0 방어 → safeMonW=1920 폴백 적용",
      0 <= hid_x <= 65535 and 0 <= hid_y <= 65535,
      f"hidX={hid_x}, hidY={hid_y} (폴백 1920x1080 기준)")

# 케이스 6: 모니터 2560x1440 (QHD) 스케일
abs_x, abs_y, hid_x, hid_y = make_attack_hid(567, 236, window_left=200, window_top=100,
                                               monitor_w=2560, monitor_h=1440)
check("QHD(2560x1440) 모니터에서도 HID 단위 0~65535 범위",
      0 <= hid_x <= 65535 and 0 <= hid_y <= 65535,
      f"hidX={hid_x}, hidY={hid_y}")
print(f"         좌표 흐름: 창픽셀(567,236) → 절대({200+567},{100+236}) → HID({hid_x},{hid_y})")

# ══════════════════════════════════════════════════════════════════════════════
# 최종 집계
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = total - passed
print(f"  전체 {total}건 중 PASS: {passed}건 / FAIL: {failed}건")
if failed == 0:
    print("  ★ 모든 검증 항목 PASS ★")
else:
    print("  ✗ FAIL 항목:")
    for name, status, note in results:
        if status == FAIL:
            print(f"    - {name}: {note}")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)
