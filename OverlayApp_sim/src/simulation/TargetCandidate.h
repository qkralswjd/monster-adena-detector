#pragma once
// TargetCandidate.h
// =================
// Detection → TargetCandidate 변환 구조체.
// 기존 Detection.h / TrackedObject.h 수정 없이 시뮬레이션 레이어에서만 사용.

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "../detection/Detection.h"
#include "../tracking/TrackedObject.h"

// ── TargetCandidate ──────────────────────────────────────────────────────────
struct TargetCandidate {
    int         id          = -1;        // 후보 고유 ID
    int         classId     = -1;
    std::string className;
    float       confidence  = 0.0f;

    // bbox (normalized [0,1], 기존 Detection 과 동일 규격)
    float       bboxX       = 0.0f;
    float       bboxY       = 0.0f;
    float       bboxW       = 0.0f;
    float       bboxH       = 0.0f;

    // 화면 픽셀 중심 (스케일 적용 후)
    float       centerPixelX = 0.0f;
    float       centerPixelY = 0.0f;

    double      detectedAtMs = 0.0;
    bool        isSelected  = false;   // 공격 대상으로 선정됐는지

    // Detection → TargetCandidate 변환
    // frameW, frameH: 캡처 해상도 (픽셀)
    static TargetCandidate FromDetection(const Detection& det,
                                         int frameW, int frameH,
                                         double nowMs,
                                         int id = -1)
    {
        TargetCandidate tc;
        tc.id            = id;
        tc.classId       = det.classId;
        tc.className     = det.className;
        tc.confidence    = det.confidence;
        tc.bboxX         = det.x;
        tc.bboxY         = det.y;
        tc.bboxW         = det.width;
        tc.bboxH         = det.height;
        tc.centerPixelX  = (det.x + det.width  * 0.5f) * static_cast<float>(frameW);
        tc.centerPixelY  = (det.y + det.height * 0.5f) * static_cast<float>(frameH);
        tc.detectedAtMs  = nowMs;
        return tc;
    }

    // TrackedObject → TargetCandidate 변환
    static TargetCandidate FromTrackedObject(const TrackedObject& obj,
                                              int frameW, int frameH,
                                              double nowMs)
    {
        TargetCandidate tc;
        tc.id            = obj.trackId;
        tc.classId       = obj.classId;
        tc.className     = obj.className;
        tc.confidence    = obj.confidence;
        tc.bboxX         = obj.bbox.x;
        tc.bboxY         = obj.bbox.y;
        tc.bboxW         = obj.bbox.width;
        tc.bboxH         = obj.bbox.height;
        tc.centerPixelX  = obj.center.x * static_cast<float>(frameW);
        tc.centerPixelY  = obj.center.y * static_cast<float>(frameH);
        tc.detectedAtMs  = nowMs;
        return tc;
    }

    std::string ToJson() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{"
           << "\"id\":"          << id          << ","
           << "\"classId\":"     << classId     << ","
           << "\"className\":\"" << className   << "\","
           << "\"confidence\":"  << confidence  << ","
           << "\"bboxX\":"       << bboxX       << ","
           << "\"bboxY\":"       << bboxY       << ","
           << "\"bboxW\":"       << bboxW       << ","
           << "\"bboxH\":"       << bboxH       << ","
           << "\"centerX\":"     << centerPixelX << ","
           << "\"centerY\":"     << centerPixelY << ","
           << "\"detectedAtMs\":" << std::setprecision(2) << detectedAtMs << ","
           << "\"isSelected\":"  << (isSelected ? "true" : "false")
           << "}";
        return ss.str();
    }
};

// ── 공격 시뮬레이션 이벤트 ───────────────────────────────────────────────────
struct AttackSimEvent {
    int         targetId      = -1;
    float       targetCenterX = 0.0f;
    float       targetCenterY = 0.0f;
    float       confidence    = 0.0f;
    double      startMs       = 0.0;
    double      endMs         = 0.0;

    // 시뮬레이션된 drag 정보 (실제 입력 없음)
    // dragFromX/Y, dragToX/Y = Pico HID 단위 (0~65535)
    // absFromX/Y, absToX/Y   = 모니터 절대 픽셀 (디버그용)
    int         dragFromX     = 0;   // Pico HID 단위
    int         dragFromY     = 0;
    int         dragToX       = 0;   // Pico HID 단위
    int         dragToY       = 0;
    double      dragDurationMs= 150.0;

    std::string ToJson() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{"
           << "\"event\":\"AttackSim\","
           << "\"targetId\":"      << targetId      << ","
           << "\"targetCenterX\":" << targetCenterX << ","
           << "\"targetCenterY\":" << targetCenterY << ","
           << "\"confidence\":"    << std::setprecision(4) << confidence << ","
           << "\"startMs\":"       << std::setprecision(2) << startMs   << ","
           << "\"endMs\":"         << endMs         << ","
           << "\"dragFromX_hid\":" << dragFromX     << ","
           << "\"dragFromY_hid\":" << dragFromY     << ","
           << "\"dragToX_hid\":"   << dragToX       << ","
           << "\"dragToY_hid\":"   << dragToY       << ","
           << "\"dragDurationMs\":" << dragDurationMs
           << "}";
        return ss.str();
    }
};

// ── 이동 시뮬레이션 이벤트 ───────────────────────────────────────────────────
struct MoveSimEvent {
    std::string fromPoint;
    std::string toPoint;
    float       fromX     = 0.0f;
    float       fromY     = 0.0f;
    float       toX       = 0.0f;
    float       toY       = 0.0f;
    double      startMs   = 0.0;
    double      arrivalMs = 0.0;
    bool        arrived   = false;

    std::string ToJson() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{"
           << "\"event\":\"MoveSim\","
           << "\"fromPoint\":\"" << fromPoint << "\","
           << "\"toPoint\":\""   << toPoint   << "\","
           << "\"fromX\":"       << fromX     << ","
           << "\"fromY\":"       << fromY     << ","
           << "\"toX\":"         << toX       << ","
           << "\"toY\":"         << toY       << ","
           << "\"startMs\":"     << startMs   << ","
           << "\"arrivalMs\":"   << arrivalMs << ","
           << "\"arrived\":"     << (arrived ? "true" : "false")
           << "}";
        return ss.str();
    }
};
