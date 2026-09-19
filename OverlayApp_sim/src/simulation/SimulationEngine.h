#pragma once
// SimulationEngine.h
// ==================
// 오프라인/시뮬레이션 전용 파이프라인.
//
// 흐름:
//   Detection  →  TargetCandidate  →  SelectTarget  →  AttackSimEvent  → MockPicoHID
//   HuntingGround → Waypoint → MoveSimEvent → MockPicoHID
//
// 실제 마우스/키보드/HID 장치에 절대 전달하지 않음.
// 기존 YoloDetector, TrackedObject, TargetManager 코드 수정 없음.

#include "MockPicoHID.h"
#include "TargetCandidate.h"
#include "HuntingGround.h"

#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

// ── 시뮬레이션 설정 ──────────────────────────────────────────────────────────
struct SimEngineConfig {
    // 프레임 해상도 (Detection 정규화 좌표 → 게임 창 픽셀 변환에 사용)
    // ScreenCapture가 캡처한 게임 창 전체 크기 (ROI 아님)
    int   frameW          = 1135;   // 게임 창 너비 (픽셀)
    int   frameH          = 472;    // 게임 창 높이 (픽셀)

    // ── Pico HID 절대좌표 변환용 ─────────────────────────────────────────
    // main.cpp: DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS) 결과를
    // gameWindowRect에서 읽어서 채워 넣는다.
    //
    //   SimEngineConfig cfg;
    //   cfg.windowLeft = gameWindowRect.x;   // rect.left
    //   cfg.windowTop  = gameWindowRect.y;   // rect.top
    //   cfg.monitorW   = GetSystemMetrics(SM_CXSCREEN);
    //   cfg.monitorH   = GetSystemMetrics(SM_CYSCREEN);
    //
    // 변환 공식:
    //   absX = windowLeft + centerPixelX          (모니터 절대 픽셀)
    //   hidX = absX * 65535 / monitorW            (Pico HID 단위 0~65535)
    int   windowLeft      = 0;      // 게임 창 좌상단 X (모니터 절대 픽셀)
    int   windowTop       = 0;      // 게임 창 좌상단 Y (모니터 절대 픽셀)
    int   monitorW        = 1920;   // 모니터 전체 너비 (픽셀)
    int   monitorH        = 1080;   // 모니터 전체 높이 (픽셀)

    // 공격 선택 기준
    float minConfidence   = 0.50f;  // CASE A 기준과 동일
    int   minTrackLen     = 2;      // Confirmed track 최소 길이

    // 공격 시뮬레이션
    double attackDragMs   = 150.0;  // 시뮬레이션 drag 소요 시간 (ms)

    // 이동 시뮬레이션
    double moveSpeedPxPerSec = 200.0; // 가상 이동 속도 (픽셀/초)

    // 캐릭터 초기 위치 (픽셀, 게임 창 기준)
    float  charStartX     = 567.0f; // 게임 창 중심 근처
    float  charStartY     = 236.0f;

    // 로그 레벨 (0=quiet, 1=summary, 2=verbose)
    int    logLevel       = 2;
};

// ── 시뮬레이션 결과 스냅샷 ───────────────────────────────────────────────────
struct SimSessionResult {
    int   totalFrames      = 0;
    int   framesWithDet    = 0;       // detection이 있는 프레임 수
    int   candidatesTotal  = 0;
    int   attacksSimulated = 0;
    int   movesSimulated   = 0;

    std::vector<AttackSimEvent> attackEvents;
    std::vector<MoveSimEvent>   moveEvents;

    // MockPicoHID 전체 이벤트 로그 (세션 종료 후 DrainEvents 결과)
    std::vector<MockHidEvent>   hidEvents;

    std::string ToJson() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"summary\": {\n";
        ss << "    \"totalFrames\": "     << totalFrames     << ",\n";
        ss << "    \"framesWithDet\": "   << framesWithDet   << ",\n";
        ss << "    \"candidatesTotal\": " << candidatesTotal << ",\n";
        ss << "    \"attacksSimulated\": "<< attacksSimulated<< ",\n";
        ss << "    \"movesSimulated\": "  << movesSimulated  << "\n";
        ss << "  },\n";

        // attack events
        ss << "  \"attackEvents\": [\n";
        for (size_t i = 0; i < attackEvents.size(); i++) {
            ss << "    " << attackEvents[i].ToJson();
            ss << (i + 1 < attackEvents.size() ? "," : "") << "\n";
        }
        ss << "  ],\n";

        // move events
        ss << "  \"moveEvents\": [\n";
        for (size_t i = 0; i < moveEvents.size(); i++) {
            ss << "    " << moveEvents[i].ToJson();
            ss << (i + 1 < moveEvents.size() ? "," : "") << "\n";
        }
        ss << "  ],\n";

        // hid log
        ss << "  \"hidLog\": " << HidEventsToJson(hidEvents) << "\n";
        ss << "}\n";
        return ss.str();
    }

private:
    static std::string HidEventsToJson(const std::vector<MockHidEvent>& evs) {
        std::ostringstream ss;
        ss << "[\n";
        for (size_t i = 0; i < evs.size(); i++) {
            const auto& e = evs[i];
            ss << "    {\n";
            ss << "      \"type\": \""      << MockHidEventTypeName(e.type) << "\",\n";
            ss << "      \"timestampMs\": " << std::fixed << std::setprecision(2) << e.timestampMs << ",\n";
            ss << "      \"x\": "           << e.x   << ", \"y\": " << e.y << ",\n";
            ss << "      \"toX\": "         << e.toX << ", \"toY\": " << e.toY << ",\n";
            ss << "      \"button\": "      << e.button << ", \"keyCode\": " << e.keyCode << ",\n";
            ss << "      \"durationMs\": "  << e.durationMs << ",\n";
            ss << "      \"tag\": \""       << e.tag << "\"\n";
            ss << "    }" << (i + 1 < evs.size() ? "," : "") << "\n";
        }
        ss << "  ]";
        return ss.str();
    }
};

// ── SimulationEngine ─────────────────────────────────────────────────────────
class SimulationEngine {
public:
    // ── 생성 / 설정 ──────────────────────────────────────────────────────────
    explicit SimulationEngine(const SimEngineConfig& cfg = SimEngineConfig{})
        : m_cfg(cfg)
        , m_charX(cfg.charStartX)
        , m_charY(cfg.charStartY)
        , m_nextCandidateId(0)
        , m_sessionStartMs(NowMs())
    {
        // MockPicoHID 콜백: 모든 이벤트를 즉시 콘솔 출력 (logLevel>=2)
        MockPicoHID::Instance().SetOnEvent([this](const MockHidEvent& ev) {
            if (m_cfg.logLevel >= 2) {
                std::printf("  [MockHID] %s  t=%.1fms  (%d,%d)→(%d,%d)  btn=%d key=%d dur=%.0fms  tag=%s\n",
                    MockHidEventTypeName(ev.type),
                    ev.timestampMs,
                    ev.x, ev.y, ev.toX, ev.toY,
                    ev.button, ev.keyCode,
                    ev.durationMs,
                    ev.tag.c_str());
            }
        });
    }

    // 설정 변경
    void SetConfig(const SimEngineConfig& cfg) { m_cfg = cfg; }
    const SimEngineConfig& GetConfig() const    { return m_cfg; }

    // 캐릭터 현재 위치 조회
    float CharX() const { return m_charX; }
    float CharY() const { return m_charY; }

    // ── 핵심 API ─────────────────────────────────────────────────────────────

    // ① 한 프레임의 Detection 목록으로 TargetCandidate 생성 + 선택 + 공격 시뮬
    //    반환값: 이 프레임에서 생성된 AttackSimEvent 목록 (없으면 empty)
    std::vector<AttackSimEvent> ProcessDetections(
        const std::vector<Detection>& detections,
        double nowMs = -1.0)
    {
        if (nowMs < 0.0) nowMs = ElapsedMs();
        m_result.totalFrames++;

        // 후보 생성
        std::vector<TargetCandidate> candidates;
        candidates.reserve(detections.size());
        for (auto& det : detections) {
            if (det.confidence < m_cfg.minConfidence) continue;
            TargetCandidate tc = TargetCandidate::FromDetection(
                det, m_cfg.frameW, m_cfg.frameH, nowMs, m_nextCandidateId++);
            candidates.push_back(tc);
            m_result.candidatesTotal++;
        }

        if (!detections.empty())
            m_result.framesWithDet++;

        if (candidates.empty())
            return {};

        if (m_cfg.logLevel >= 1)
            std::printf("[SimEngine] Frame %d: %zu detection(s), %zu candidate(s)\n",
                m_result.totalFrames, detections.size(), candidates.size());

        // 목표 선택 (center 우선: 화면 중심에 가장 가까운 후보)
        TargetCandidate* selected = SelectTarget(candidates);
        if (!selected) return {};
        selected->isSelected = true;

        // 공격 이벤트 생성
        AttackSimEvent ev = MakeAttackEvent(*selected, nowMs);
        m_result.attackEvents.push_back(ev);
        m_result.attacksSimulated++;

        // MockPicoHID에 drag 기록 (실제 입력 없음)
        MockPicoHID::Instance().Drag(
            ev.dragFromX, ev.dragFromY,
            ev.dragToX,   ev.dragToY,
            ev.dragDurationMs,
            "AttackSim:target_" + std::to_string(ev.targetId)
        );

        if (m_cfg.logLevel >= 1)
            std::printf("[SimEngine] Attack  → target#%d  center(%.1f,%.1f)  conf=%.3f\n",
                ev.targetId, ev.targetCenterX, ev.targetCenterY, ev.confidence);

        return { ev };
    }

    // ② TrackedObject 목록으로 TargetCandidate 생성 + 공격 시뮬
    std::vector<AttackSimEvent> ProcessTrackedObjects(
        const std::vector<TrackedObject>& objects,
        double nowMs = -1.0)
    {
        if (nowMs < 0.0) nowMs = ElapsedMs();
        m_result.totalFrames++;

        std::vector<TargetCandidate> candidates;
        candidates.reserve(objects.size());
        for (auto& obj : objects) {
            if (obj.state != TrackState::Confirmed)    continue;
            if (obj.confidence < m_cfg.minConfidence)  continue;
            if (obj.hitStreak  < m_cfg.minTrackLen)    continue;
            TargetCandidate tc = TargetCandidate::FromTrackedObject(
                obj, m_cfg.frameW, m_cfg.frameH, nowMs);
            candidates.push_back(tc);
            m_result.candidatesTotal++;
        }

        if (!objects.empty()) m_result.framesWithDet++;
        if (candidates.empty()) return {};

        if (m_cfg.logLevel >= 1)
            std::printf("[SimEngine] Tracked Frame %d: %zu object(s), %zu candidate(s)\n",
                m_result.totalFrames, objects.size(), candidates.size());

        TargetCandidate* selected = SelectTarget(candidates);
        if (!selected) return {};
        selected->isSelected = true;

        AttackSimEvent ev = MakeAttackEvent(*selected, nowMs);
        m_result.attackEvents.push_back(ev);
        m_result.attacksSimulated++;

        MockPicoHID::Instance().Drag(
            ev.dragFromX, ev.dragFromY,
            ev.dragToX,   ev.dragToY,
            ev.dragDurationMs,
            "AttackSim:track_" + std::to_string(ev.targetId)
        );

        if (m_cfg.logLevel >= 1)
            std::printf("[SimEngine] Attack  → track#%d  center(%.1f,%.1f)  conf=%.3f\n",
                ev.targetId, ev.targetCenterX, ev.targetCenterY, ev.confidence);

        return { ev };
    }

    // ③ HuntingGround의 특정 루트를 웨이포인트 순서대로 이동 시뮬레이션
    //    반환값: 생성된 MoveSimEvent 목록
    std::vector<MoveSimEvent> SimulateRoute(
        const HuntingGround& hg,
        int routeIndex = 0,
        double startMs = -1.0)
    {
        if (startMs < 0.0) startMs = ElapsedMs();
        std::vector<MoveSimEvent> events;

        if (routeIndex < 0 || routeIndex >= static_cast<int>(hg.routes.size())) {
            std::printf("[SimEngine] ERROR: routeIndex %d out of range (routes=%zu)\n",
                routeIndex, hg.routes.size());
            return events;
        }

        const auto& route = hg.routes[routeIndex];
        if (route.size() < 2) return events;

        double curMs = startMs;
        float  curX  = m_charX;
        float  curY  = m_charY;

        if (m_cfg.logLevel >= 1)
            std::printf("[SimEngine] Route[%d] start: char(%.0f,%.0f)  waypoints=%zu\n",
                routeIndex, curX, curY, route.size());

        for (size_t i = 0; i + 1 < route.size(); i++) {
            const std::string& fromName = route[i];
            const std::string& toName   = route[i + 1];

            const Waypoint* wp = hg.FindPoint(toName);
            if (!wp) {
                std::printf("[SimEngine] WARNING: waypoint '%s' not found, skipping\n", toName.c_str());
                continue;
            }

            float dist = std::sqrt(std::pow(wp->x - curX, 2.0f) + std::pow(wp->y - curY, 2.0f));
            double travelMs = (m_cfg.moveSpeedPxPerSec > 0.0)
                ? (dist / m_cfg.moveSpeedPxPerSec) * 1000.0
                : 0.0;

            MoveSimEvent mv;
            mv.fromPoint  = fromName;
            mv.toPoint    = toName;
            mv.fromX      = curX;
            mv.fromY      = curY;
            mv.toX        = wp->x;
            mv.toY        = wp->y;
            mv.startMs    = curMs;
            mv.arrivalMs  = curMs + travelMs;
            mv.arrived    = true;

            // MockPicoHID에 이동 기록 (실제 입력 없음)
            MockPicoHID::Instance().Move(
                static_cast<int>(curX), static_cast<int>(curY),
                "MoveSim:from_" + fromName
            );
            MockPicoHID::Instance().Delay(travelMs, "MoveSim:travel_" + fromName + "_to_" + toName);
            MockPicoHID::Instance().Move(
                static_cast<int>(wp->x), static_cast<int>(wp->y),
                "MoveSim:arrive_" + toName
            );

            events.push_back(mv);
            m_result.moveEvents.push_back(mv);
            m_result.movesSimulated++;

            if (m_cfg.logLevel >= 1)
                std::printf("[SimEngine] Move   %s → %s  dist=%.1fpx  travel=%.0fms\n",
                    fromName.c_str(), toName.c_str(), dist, travelMs);

            curX  = wp->x;
            curY  = wp->y;
            curMs = mv.arrivalMs;
        }

        // 캐릭터 위치 업데이트
        m_charX = curX;
        m_charY = curY;

        return events;
    }

    // ④ 세션 종료 → 결과 수집
    SimSessionResult FinalizeSession() {
        // MockPicoHID 전체 이벤트 수집 (DrainEvents 는 큐를 비움)
        m_result.hidEvents = MockPicoHID::Instance().DrainEvents();
        return m_result;
    }

    // 세션 초기화 (재사용)
    void ResetSession() {
        m_result        = SimSessionResult{};
        m_nextCandidateId = 0;
        m_sessionStartMs  = NowMs();
        m_charX = m_cfg.charStartX;
        m_charY = m_cfg.charStartY;
        MockPicoHID::Instance().ClearHistory();
    }

    // 경과 시간 (세션 시작 기준 ms)
    double ElapsedMs() const {
        return NowMs() - m_sessionStartMs;
    }

private:
    // ── 내부 유틸 ────────────────────────────────────────────────────────────

    // 목표 선택: 화면 중심에 가장 가까운 Confirmed 후보
    TargetCandidate* SelectTarget(std::vector<TargetCandidate>& candidates) {
        if (candidates.empty()) return nullptr;
        float cx = static_cast<float>(m_cfg.frameW) * 0.5f;
        float cy = static_cast<float>(m_cfg.frameH) * 0.5f;

        TargetCandidate* best = nullptr;
        float bestDist = std::numeric_limits<float>::max();
        for (auto& tc : candidates) {
            float dx = tc.centerPixelX - cx;
            float dy = tc.centerPixelY - cy;
            float d  = dx * dx + dy * dy;
            if (d < bestDist) { bestDist = d; best = &tc; }
        }
        return best;
    }

    // 공격 이벤트 생성
    // ── 좌표 변환 파이프라인 ─────────────────────────────────────────────
    // 1) tc.centerPixelX/Y : 게임 창 픽셀 (0 ~ frameW/frameH)
    // 2) 모니터 절대 픽셀  : windowLeft + centerPixelX
    // 3) Pico HID 단위     : absX * 65535 / monitorW  (0 ~ 65535)
    //
    // windowLeft/Top = DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)
    // monitorW/H     = GetSystemMetrics(SM_CXSCREEN / SM_CYSCREEN)
    // ─────────────────────────────────────────────────────────────────────
    AttackSimEvent MakeAttackEvent(const TargetCandidate& tc, double nowMs) {
        AttackSimEvent ev;
        ev.targetId       = tc.id;
        ev.targetCenterX  = tc.centerPixelX;
        ev.targetCenterY  = tc.centerPixelY;
        ev.confidence     = tc.confidence;
        ev.startMs        = nowMs;
        ev.endMs          = nowMs + m_cfg.attackDragMs;

        // ── 게임 창 픽셀 → 모니터 절대 픽셀 ──────────────────────────
        const int absTargetX = m_cfg.windowLeft + static_cast<int>(tc.centerPixelX);
        const int absTargetY = m_cfg.windowTop  + static_cast<int>(tc.centerPixelY);
        const int absCharX   = m_cfg.windowLeft + static_cast<int>(m_charX);
        const int absCharY   = m_cfg.windowTop  + static_cast<int>(m_charY);

        // ── 모니터 절대 픽셀 → Pico HID 단위 (0~65535) ───────────────
        // monitorW/H가 0이면 변환 없이 픽셀 그대로 (실수 방지 가드)
        const int safeMonW = (m_cfg.monitorW > 0) ? m_cfg.monitorW : 1920;
        const int safeMonH = (m_cfg.monitorH > 0) ? m_cfg.monitorH : 1080;

        ev.dragFromX      = absCharX   * 65535 / safeMonW;
        ev.dragFromY      = absCharY   * 65535 / safeMonH;
        ev.dragToX        = absTargetX * 65535 / safeMonW;
        ev.dragToY        = absTargetY * 65535 / safeMonH;
        ev.dragDurationMs = m_cfg.attackDragMs;

        if (m_cfg.logLevel >= 2) {
            std::printf(
                "  [CoordConv] gameWinPx(%.0f,%.0f)"
                " -> absPx(%d,%d)"
                " -> hidUnit(%d,%d)\n",
                tc.centerPixelX, tc.centerPixelY,
                absTargetX, absTargetY,
                ev.dragToX, ev.dragToY
            );
        }
        return ev;
    }

    static double NowMs() {
        using namespace std::chrono;
        return duration<double, std::milli>(
            steady_clock::now().time_since_epoch()
        ).count();
    }

    SimEngineConfig   m_cfg;
    float             m_charX;
    float             m_charY;
    int               m_nextCandidateId;
    double            m_sessionStartMs;
    SimSessionResult  m_result;
};
