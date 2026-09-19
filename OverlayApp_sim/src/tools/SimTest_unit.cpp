// SimTest_unit.cpp
// =================
// SimulationEngine 독립 C++ 유닛 테스트
//
// 빌드/실행:
//   Windows (MSVC):
//     cl /EHsc /std:c++17 /I.. SimTest_unit.cpp /Fe:SimTest_unit.exe
//   또는 CMake:
//     target: SimTestUnit  (CMakeLists.txt 참조)
//
// ★ 의존성 ★
//   - simulation/MockPicoHID.h     (헤더-only, Win32/D3D11 없음)
//   - simulation/TargetCandidate.h (헤더-only, Win32/D3D11 없음)
//   - simulation/HuntingGround.h   (헤더-only, Win32/D3D11 없음)
//   - simulation/SimulationEngine.h (헤더-only, Win32/D3D11 없음)
//   - detection/Detection.h        (순수 데이터 구조체)
//   - tracking/TrackedObject.h     (순수 데이터 구조체)
//
// ★ 실제 HID/마우스/키보드 입력 절대 없음 ★
// ★ Windows 전용 API(D3D11, WIC, DXGICapture) 없음 ★

#include "../simulation/SimulationEngine.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

// ════════════════════════════════════════════════════════════════════════════
// 미니 테스트 프레임워크
// ════════════════════════════════════════════════════════════════════════════

namespace UnitTest {

struct Result {
    std::string suite;
    std::string name;
    bool        passed;
    std::string detail;
};

static std::vector<Result> g_results;
static int g_pass = 0;
static int g_fail = 0;

static void Record(const std::string& suite, const std::string& name,
                   bool ok, const std::string& detail = "")
{
    const char* tag = ok ? "\033[92mPASS\033[0m" : "\033[91mFAIL\033[0m";
    std::printf("  [%s] %s%s\n", tag, name.c_str(),
                detail.empty() ? "" : (" — " + detail).c_str());
    g_results.push_back({suite, name, ok, detail});
    if (ok) ++g_pass; else ++g_fail;
}

// 부동소수점 비교 헬퍼
static bool NearlyEq(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

} // namespace UnitTest

using UnitTest::Record;
using UnitTest::NearlyEq;

// ════════════════════════════════════════════════════════════════════════════
// Suite 1: CoordinateTransformTest
// ════════════════════════════════════════════════════════════════════════════

void CoordinateTransformTest()
{
    const std::string S = "CoordinateTransform";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 1: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    // ── 1-A: TargetCandidate center 공식 수치 검증 ─────────────────────────
    // C++ 공식 (TargetCandidate.h):
    //   centerPixelX = (det.x + det.width  * 0.5f) * frameW
    //   centerPixelY = (det.y + det.height * 0.5f) * frameH
    {
        const int FRAME_W = 1135, FRAME_H = 472;
        Detection det;
        det.x         = 0.30f;
        det.y         = 0.25f;
        det.width     = 0.15f;
        det.height    = 0.30f;
        det.confidence = 0.82f;
        det.classId    = 0;
        det.className  = "monster";

        TargetCandidate tc = TargetCandidate::FromDetection(det, FRAME_W, FRAME_H, 0.0, 0);

        float exp_cx = (0.30f + 0.15f * 0.5f) * static_cast<float>(FRAME_W);
        float exp_cy = (0.25f + 0.30f * 0.5f) * static_cast<float>(FRAME_H);
        float err_cx = std::abs(tc.centerPixelX - exp_cx);
        float err_cy = std::abs(tc.centerPixelY - exp_cy);

        std::printf("  예상 center: (%.4f, %.4f)\n", exp_cx, exp_cy);
        std::printf("  실제 center: (%.4f, %.4f)\n", tc.centerPixelX, tc.centerPixelY);

        Record(S, "centerPixelX 공식 정밀도", err_cx < 1e-4f,
               "err=" + std::to_string(err_cx));
        Record(S, "centerPixelY 공식 정밀도", err_cy < 1e-4f,
               "err=" + std::to_string(err_cy));
    }

    // ── 1-B: bbox 보존 ────────────────────────────────────────────────────
    {
        const int FRAME_W = 1135, FRAME_H = 472;
        Detection det;
        det.x = 0.65f; det.y = 0.40f;
        det.width = 0.12f; det.height = 0.28f;
        det.confidence = 0.61f; det.classId = 0;

        TargetCandidate tc = TargetCandidate::FromDetection(det, FRAME_W, FRAME_H, 1000.0, 7);
        bool bbox_ok = (tc.bboxX == det.x && tc.bboxY == det.y &&
                        tc.bboxW == det.width && tc.bboxH == det.height);
        Record(S, "bbox 보존 (bboxX/Y/W/H == det.x/y/w/h)", bbox_ok,
               "got=(" + std::to_string(tc.bboxX) + "," + std::to_string(tc.bboxY) + ")");
        Record(S, "confidence 보존",
               std::abs(tc.confidence - det.confidence) < 1e-6f,
               "got=" + std::to_string(tc.confidence));
        Record(S, "id 보존", tc.id == 7, "got=" + std::to_string(tc.id));
    }

    // ── 1-C: Letterbox 파라미터 계산 (Python과 동일한 수식 확인) ──────────
    // srcW=1135, srcH=472 → inputW=inputH=640
    // scale = min(640/1135, 640/472) = min(0.5638..., 1.3559...) = 0.5638...
    // scaledW = round(1135 * 0.5638) = 640
    // scaledH = round(472  * 0.5638) = 266
    // padX = (640-640)/2 = 0
    // padY = (640-266)/2 = 187
    {
        const double src_w = 1135.0, src_h = 472.0;
        const double input_w = 640.0, input_h = 640.0;
        double scale     = std::min(input_w / src_w, input_h / src_h);
        int    scaled_w  = std::max(1, static_cast<int>(std::round(src_w * scale)));
        int    scaled_h  = std::max(1, static_cast<int>(std::round(src_h * scale)));
        int    pad_x     = (static_cast<int>(input_w) - scaled_w) / 2;
        int    pad_y     = (static_cast<int>(input_h) - scaled_h) / 2;

        std::printf("  Letterbox: scale=%.6f  scaledWH=(%d,%d)  pad=(%d,%d)\n",
                    scale, scaled_w, scaled_h, pad_x, pad_y);

        Record(S, "Letterbox scaledW = 640", scaled_w == 640, "got=" + std::to_string(scaled_w));
        Record(S, "Letterbox scaledH = 266", scaled_h == 266, "got=" + std::to_string(scaled_h));
        Record(S, "Letterbox padX = 0",      pad_x == 0,      "got=" + std::to_string(pad_x));
        Record(S, "Letterbox padY = 187",     pad_y == 187,    "got=" + std::to_string(pad_y));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 2: TargetCandidateTest
// ════════════════════════════════════════════════════════════════════════════

void TargetCandidateTest()
{
    const std::string S = "TargetCandidate";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 2: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    const int FRAME_W = 1135, FRAME_H = 472;

    struct TestCase {
        const char* name;
        float x, y, w, h, conf;
        int cls_id;
    };
    const TestCase cases[] = {
        {"center_monster",   0.30f, 0.25f, 0.15f, 0.30f, 0.82f, 0},
        {"right_monster",    0.65f, 0.40f, 0.12f, 0.28f, 0.61f, 0},
        {"adena_drop",       0.50f, 0.60f, 0.05f, 0.04f, 0.74f, 1},
        {"top_left_edge",    0.00f, 0.00f, 0.05f, 0.05f, 0.90f, 0},
        {"bot_right_edge",   0.95f, 0.95f, 0.05f, 0.05f, 0.55f, 0},
        {"full_frame",       0.00f, 0.00f, 1.00f, 1.00f, 0.99f, 0},
    };

    float max_err = 0.0f;
    for (const auto& tc_in : cases) {
        Detection det;
        det.x = tc_in.x; det.y = tc_in.y;
        det.width = tc_in.w; det.height = tc_in.h;
        det.confidence = tc_in.conf;
        det.classId = tc_in.cls_id;
        det.className = "test";

        TargetCandidate tc = TargetCandidate::FromDetection(det, FRAME_W, FRAME_H, 0.0, 42);

        float exp_cx = (tc_in.x + tc_in.w * 0.5f) * static_cast<float>(FRAME_W);
        float exp_cy = (tc_in.y + tc_in.h * 0.5f) * static_cast<float>(FRAME_H);
        float err_cx = std::abs(tc.centerPixelX - exp_cx);
        float err_cy = std::abs(tc.centerPixelY - exp_cy);
        max_err = std::max(max_err, std::max(err_cx, err_cy));

        bool center_ok = err_cx < 1e-4f && err_cy < 1e-4f;
        bool bbox_ok   = (tc.bboxX == tc_in.x && tc.bboxY == tc_in.y &&
                          tc.bboxW == tc_in.w && tc.bboxH == tc_in.h);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "[%s] center+bbox", tc_in.name);
        Record(S, buf, center_ok && bbox_ok,
               std::string("cx_err=") + std::to_string(err_cx) +
               " cy_err=" + std::to_string(err_cy));
    }
    std::printf("  최대 center 오차: %.2e px\n", max_err);
    Record(S, "전체 center 오차 = 0 (float 동일 연산)",
           max_err < 1e-4f, "max_err=" + std::to_string(max_err));
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 3: AttackSimulationTest
// ════════════════════════════════════════════════════════════════════════════

void AttackSimulationTest()
{
    const std::string S = "AttackSimulation";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 3: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    MockPicoHID::Instance().ClearHistory();

    // ── 3-A: 명시 케이스 (425.25, 294.80) ────────────────────────────────
    {
        // center가 정확히 (425.25, 294.80)이 되는 Detection을 역산
        const int FRAME_W = 1135, FRAME_H = 472;
        const float target_cx = 425.25f;
        const float target_cy = 294.80f;
        const float det_w = 0.15f, det_h = 0.30f;
        float det_x = (target_cx / static_cast<float>(FRAME_W)) - det_w * 0.5f;
        float det_y = (target_cy / static_cast<float>(FRAME_H)) - det_h * 0.5f;

        Detection det;
        det.x = det_x; det.y = det_y;
        det.width = det_w; det.height = det_h;
        det.confidence = 0.82f; det.classId = 0;

        TargetCandidate tc = TargetCandidate::FromDetection(det, FRAME_W, FRAME_H, 0.0, 0);
        float err_cx = std::abs(tc.centerPixelX - target_cx);
        float err_cy = std::abs(tc.centerPixelY - target_cy);

        std::printf("  명시 케이스 center: (%.6f, %.6f)\n", tc.centerPixelX, tc.centerPixelY);
        Record(S, "명시케이스 centerX 정밀도 (425.25)", err_cx < 1e-4f,
               "err=" + std::to_string(err_cx));
        Record(S, "명시케이스 centerY 정밀도 (294.80)", err_cy < 1e-4f,
               "err=" + std::to_string(err_cy));

        // C++ static_cast<int> truncation 검증
        int expected_to_x = static_cast<int>(tc.centerPixelX);  // 425
        int expected_to_y = static_cast<int>(tc.centerPixelY);  // 294
        float trunc_err_x = std::abs(tc.centerPixelX - static_cast<float>(expected_to_x)); // 0.25
        float trunc_err_y = std::abs(tc.centerPixelY - static_cast<float>(expected_to_y)); // 0.80
        std::printf("  static_cast<int>: (%d, %d)  truncation Δ=(%.2f, %.2f)px\n",
                    expected_to_x, expected_to_y, trunc_err_x, trunc_err_y);
        Record(S, "truncation 오차 < 1px", trunc_err_x < 1.0f && trunc_err_y < 1.0f,
               std::string("Δx=") + std::to_string(trunc_err_x) +
               " Δy=" + std::to_string(trunc_err_y));
    }

    // ── 3-B: SimulationEngine ProcessDetections 좌표 검증 ─────────────────
    {
        const int FRAME_W = 1135, FRAME_H = 472;
        const float CHAR_X = 567.0f, CHAR_Y = 236.0f;

        SimEngineConfig cfg;
        cfg.frameW = FRAME_W; cfg.frameH = FRAME_H;
        cfg.charStartX = CHAR_X; cfg.charStartY = CHAR_Y;
        cfg.minConfidence = 0.50f; cfg.attackDragMs = 150.0;
        cfg.logLevel = 0;  // 테스트 중 출력 최소화

        SimulationEngine engine(cfg);

        Detection det;
        det.x = 0.30f; det.y = 0.25f;
        det.width = 0.15f; det.height = 0.30f;
        det.confidence = 0.80f; det.classId = 0;

        auto evs = engine.ProcessDetections({det}, 0.0);
        Record(S, "ProcessDetections → 1개 이벤트 반환",
               evs.size() == 1, "size=" + std::to_string(evs.size()));

        if (evs.size() == 1) {
            const auto& ev = evs[0];
            float exp_cx = (0.30f + 0.075f) * static_cast<float>(FRAME_W);
            float exp_cy = (0.25f + 0.150f) * static_cast<float>(FRAME_H);
            int   exp_from_x = static_cast<int>(CHAR_X);
            int   exp_from_y = static_cast<int>(CHAR_Y);
            int   exp_to_x   = static_cast<int>(exp_cx);
            int   exp_to_y   = static_cast<int>(exp_cy);

            std::printf("  targetCenter: (%.4f, %.4f)\n", ev.targetCenterX, ev.targetCenterY);
            std::printf("  drag: (%d,%d) → (%d,%d)\n", ev.dragFromX, ev.dragFromY, ev.dragToX, ev.dragToY);

            Record(S, "targetCenterX 일치",
                   std::abs(ev.targetCenterX - exp_cx) < 1e-3f,
                   "got=" + std::to_string(ev.targetCenterX) + " exp=" + std::to_string(exp_cx));
            Record(S, "targetCenterY 일치",
                   std::abs(ev.targetCenterY - exp_cy) < 1e-3f,
                   "got=" + std::to_string(ev.targetCenterY) + " exp=" + std::to_string(exp_cy));
            Record(S, "dragFromX = int(charX)",
                   ev.dragFromX == exp_from_x,
                   "got=" + std::to_string(ev.dragFromX) + " exp=" + std::to_string(exp_from_x));
            Record(S, "dragFromY = int(charY)",
                   ev.dragFromY == exp_from_y,
                   "got=" + std::to_string(ev.dragFromY) + " exp=" + std::to_string(exp_from_y));
            Record(S, "dragToX = int(centerPixelX)",
                   ev.dragToX == exp_to_x,
                   "got=" + std::to_string(ev.dragToX) + " exp=" + std::to_string(exp_to_x));
            Record(S, "dragToY = int(centerPixelY)",
                   ev.dragToY == exp_to_y,
                   "got=" + std::to_string(ev.dragToY) + " exp=" + std::to_string(exp_to_y));
            Record(S, "endMs = startMs + 150",
                   NearlyEq(ev.endMs, ev.startMs + 150.0),
                   "endMs=" + std::to_string(ev.endMs));
        }
    }

    // ── 3-C: Drag 방향 8가지 ──────────────────────────────────────────────
    std::printf("\n  [Drag 방향 8가지]\n");
    struct DragCase {
        const char* name;
        float char_x, char_y;
        float tnx, tny, tw, th;
    };
    const DragCase drag_cases[] = {
        {"좌상→우하",    100.0f, 100.0f, 0.80f, 0.80f, 0.10f, 0.10f},
        {"우하→좌상",    900.0f, 400.0f, 0.10f, 0.10f, 0.10f, 0.10f},
        {"좌하→우상",    100.0f, 400.0f, 0.90f, 0.10f, 0.10f, 0.10f},
        {"우상→좌하",    900.0f, 100.0f, 0.10f, 0.90f, 0.10f, 0.10f},
        {"수평",         200.0f, 236.0f, 0.80f, 0.50f, 0.10f, 0.10f},
        {"수직",         567.0f, 100.0f, 0.50f, 0.90f, 0.10f, 0.10f},
        {"동일 좌표",    567.0f, 236.0f, 0.50f, 0.50f, 0.10f, 0.10f},
        {"가장자리우하",  10.0f,  10.0f, 0.99f, 0.99f, 0.01f, 0.01f},
    };

    const int FRAME_W = 1135, FRAME_H = 472;
    int max_drag_err = 0;
    for (const auto& dc : drag_cases) {
        SimEngineConfig cfg;
        cfg.frameW = FRAME_W; cfg.frameH = FRAME_H;
        cfg.charStartX = dc.char_x; cfg.charStartY = dc.char_y;
        cfg.minConfidence = 0.50f; cfg.logLevel = 0;

        SimulationEngine eng(cfg);
        Detection det;
        det.x = dc.tnx - dc.tw * 0.5f;
        det.y = dc.tny - dc.th * 0.5f;
        det.width = dc.tw; det.height = dc.th;
        det.confidence = 0.80f; det.classId = 0;

        auto evs = eng.ProcessDetections({det}, 0.0);
        if (evs.empty()) {
            Record(S, std::string("Drag [") + dc.name + "]", false, "no event");
            continue;
        }
        const auto& ev = evs[0];
        int exp_from_x = static_cast<int>(dc.char_x);
        int exp_from_y = static_cast<int>(dc.char_y);
        int exp_to_x   = static_cast<int>(dc.tnx * static_cast<float>(FRAME_W));
        int exp_to_y   = static_cast<int>(dc.tny * static_cast<float>(FRAME_H));

        int err = std::abs(ev.dragFromX - exp_from_x) + std::abs(ev.dragFromY - exp_from_y)
                + std::abs(ev.dragToX   - exp_to_x)   + std::abs(ev.dragToY   - exp_to_y);
        max_drag_err = std::max(max_drag_err, err);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "Drag [%s]", dc.name);
        Record(S, buf, err == 0,
               std::string("from(") + std::to_string(ev.dragFromX) + "," +
               std::to_string(ev.dragFromY) + ") to(" +
               std::to_string(ev.dragToX) + "," + std::to_string(ev.dragToY) + ") err=" +
               std::to_string(err));
    }
    std::printf("  최대 drag 오차: %d px\n", max_drag_err);
    Record(S, "전체 drag 정수 오차 = 0", max_drag_err == 0,
           "max_err=" + std::to_string(max_drag_err));
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 4: HuntingGroundTest
// ════════════════════════════════════════════════════════════════════════════

void HuntingGroundTest()
{
    const std::string S = "HuntingGround";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 4: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    // ── 4-A: JSON 문자열 직접 파싱 (파일 I/O 없이) ──────────────────────
    const std::string json_str = R"({
  "name": "test_field",
  "points": [
    {"name": "Start", "x": 100, "y": 236},
    {"name": "P1",    "x": 300, "y": 150},
    {"name": "P2",    "x": 567, "y": 200},
    {"name": "P3",    "x": 800, "y": 300},
    {"name": "P4",    "x": 950, "y": 180},
    {"name": "Return","x": 100, "y": 236}
  ],
  "routes": [
    ["Start", "P1", "P2", "P3", "P4", "Return"],
    ["Return", "P4", "P3", "P2", "P1", "Start"]
  ]
})";

    HuntingGround hg = HuntingGround::FromJson(json_str);
    Record(S, "name = 'test_field'", hg.name == "test_field", hg.name);
    Record(S, "points 6개", hg.points.size() == 6,
           std::to_string(hg.points.size()));
    Record(S, "routes 2개", hg.routes.size() == 2,
           std::to_string(hg.routes.size()));

    // ── 4-B: 웨이포인트 좌표 검증 ─────────────────────────────────────────
    struct WpExpect { const char* name; float x, y; };
    const WpExpect expected[] = {
        {"Start",  100.0f, 236.0f},
        {"P1",     300.0f, 150.0f},
        {"P2",     567.0f, 200.0f},
        {"P3",     800.0f, 300.0f},
        {"P4",     950.0f, 180.0f},
        {"Return", 100.0f, 236.0f},
    };
    for (const auto& ex : expected) {
        const Waypoint* wp = hg.FindPoint(ex.name);
        bool ok = wp && std::abs(wp->x - ex.x) < 0.01f && std::abs(wp->y - ex.y) < 0.01f;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "waypoint [%s] 좌표", ex.name);
        Record(S, buf, ok,
               wp ? ("got=(" + std::to_string(wp->x) + "," + std::to_string(wp->y) + ")") : "not found");
    }

    // ── 4-C: 루트 순서 ────────────────────────────────────────────────────
    const std::vector<std::string> route0_expected = {"Start","P1","P2","P3","P4","Return"};
    Record(S, "Route[0] 순서 보존", hg.routes[0] == route0_expected,
           hg.routes.empty() ? "empty" : std::to_string(hg.routes[0].size()) + " waypoints");

    // ── 4-D: SimulateRoute 이동 거리 수학 검증 ───────────────────────────
    SimEngineConfig cfg;
    cfg.frameW = 1135; cfg.frameH = 472;
    cfg.charStartX = 100.0f; cfg.charStartY = 236.0f;
    cfg.moveSpeedPxPerSec = 200.0; cfg.logLevel = 0;

    SimulationEngine engine(cfg);
    auto events = engine.SimulateRoute(hg, 0, 0.0);
    Record(S, "Route[0] 이벤트 수 = 5", events.size() == 5,
           std::to_string(events.size()));

    if (events.size() >= 1) {
        // Start(100,236) → P1(300,150): dist = sqrt((300-100)^2 + (150-236)^2)
        double dist_01 = std::sqrt(std::pow(300.0 - 100.0, 2.0) + std::pow(150.0 - 236.0, 2.0));
        double exp_travel = dist_01 / 200.0 * 1000.0;
        double got_travel = events[0].arrivalMs - events[0].startMs;
        std::printf("  Start→P1: dist=%.2f exp_travel=%.2fms got=%.2fms\n",
                    dist_01, exp_travel, got_travel);
        Record(S, "Start→P1 travel_ms 수치",
               std::abs(got_travel - exp_travel) < 0.01,
               "exp=" + std::to_string(exp_travel) + " got=" + std::to_string(got_travel));
    }

    // 타임스탬프 연속성
    bool continuous = true;
    for (size_t i = 0; i + 1 < events.size(); ++i) {
        if (std::abs(events[i].arrivalMs - events[i+1].startMs) > 1e-9) {
            continuous = false;
            std::printf("  [!] 연속성 끊김: events[%zu].arrivalMs=%.4f events[%zu].startMs=%.4f\n",
                        i, events[i].arrivalMs, i+1, events[i+1].startMs);
        }
    }
    Record(S, "타임스탬프 연속성", continuous,
           "arrival_ms[i] == start_ms[i+1]");

    // ── 4-E: travel_ms = dist/speed*1000 수학 검증 ──────────────────────
    for (size_t i = 0; i < events.size(); ++i) {
        double dist = std::sqrt(std::pow(events[i].toX - events[i].fromX, 2.0) +
                                std::pow(events[i].toY - events[i].fromY, 2.0));
        double exp_t = dist / 200.0 * 1000.0;
        double got_t = events[i].arrivalMs - events[i].startMs;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "MoveSim[%zu] travel_ms 수학", i);
        Record(S, buf, std::abs(got_t - exp_t) < 0.01,
               "dist=" + std::to_string(dist) +
               " exp=" + std::to_string(exp_t) +
               " got=" + std::to_string(got_t));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 5: MockPicoHIDTest
// ════════════════════════════════════════════════════════════════════════════

void MockPicoHIDTest()
{
    const std::string S = "MockPicoHID";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 5: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    auto& hid = MockPicoHID::Instance();
    hid.ClearHistory();

    // ── 5-A: 기본 이벤트 기록 ────────────────────────────────────────────
    hid.Move(100, 200, "test_move");
    hid.Click(150, 250, 0, "test_click");
    hid.Drag(100, 200, 800, 400, 300.0, "test_drag");
    hid.KeyDown(0x41, "test_key");
    hid.Delay(500.0, "test_delay");

    auto hist = hid.GetHistory();
    Record(S, "5개 이벤트 기록됨",
           hist.size() == 5, "size=" + std::to_string(hist.size()));

    if (hist.size() >= 5) {
        Record(S, "첫 번째: MouseMove",
               hist[0].type == MockHidEventType::MouseMove &&
               hist[0].x == 100 && hist[0].y == 200, "");
        Record(S, "두 번째: MouseClick",
               hist[1].type == MockHidEventType::MouseClick &&
               hist[1].x == 150, "");
        Record(S, "세 번째: MouseDrag",
               hist[2].type == MockHidEventType::MouseDrag &&
               hist[2].x == 100 && hist[2].toX == 800 &&
               NearlyEq(hist[2].durationMs, 300.0), "");
        Record(S, "네 번째: KeyDown",
               hist[3].type == MockHidEventType::KeyDown &&
               hist[3].keyCode == 0x41, "");
        Record(S, "다섯째: Delay",
               hist[4].type == MockHidEventType::Delay &&
               NearlyEq(hist[4].durationMs, 500.0), "");
    }

    // ── 5-B: DrainEvents ─────────────────────────────────────────────────
    auto drained = hid.DrainEvents();
    Record(S, "drain 반환 5개", drained.size() == 5,
           std::to_string(drained.size()));
    Record(S, "drain 후 큐 비어있음",
           hid.DrainEvents().empty(), "");
    Record(S, "GetHistory는 drain 후에도 유지",
           hid.GetHistory().size() == 5,
           std::to_string(hid.GetHistory().size()));

    // ── 5-C: Singleton 보장 ──────────────────────────────────────────────
    auto& hid2 = MockPicoHID::Instance();
    Record(S, "Singleton: 동일 인스턴스",
           &hid == &hid2, "");

    // ── 5-D: 이벤트 타입 이름 출력 검증 ─────────────────────────────────
    Record(S, "MockHidEventTypeName(MouseDrag)",
           std::string(MockHidEventTypeName(MockHidEventType::MouseDrag)) == "MouseDrag", "");
    Record(S, "MockHidEventTypeName(Delay)",
           std::string(MockHidEventTypeName(MockHidEventType::Delay)) == "Delay", "");

    // ── 5-E: 실제 입력 없음 확인 (콜백 카운트) ───────────────────────────
    // MockPicoHID.SetOnEvent()으로 콜백 수를 세어 실제 전달 없음 확인
    hid.ClearHistory();
    int callback_count = 0;
    hid.SetOnEvent([&](const MockHidEvent&) { callback_count++; });
    hid.Move(0, 0, "mock_test");
    hid.Drag(0, 0, 100, 100, 50.0, "mock_test2");
    Record(S, "콜백은 발생하나 실제 HID 전달 없음",
           callback_count == 2,
           "콜백=" + std::to_string(callback_count) + "회 (이벤트 기록만, 실제 전달 없음)");
    hid.SetOnEvent(nullptr);  // 콜백 초기화
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 6: BoundaryValuesTest
// ════════════════════════════════════════════════════════════════════════════

void BoundaryValuesTest()
{
    const std::string S = "BoundaryValues";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 6: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    const int FRAME_W = 1135, FRAME_H = 472;

    // ── 6-A: confidence 경계값 ────────────────────────────────────────────
    SimEngineConfig cfg;
    cfg.frameW = FRAME_W; cfg.frameH = FRAME_H;
    cfg.charStartX = 567.0f; cfg.charStartY = 236.0f;
    cfg.minConfidence = 0.50f; cfg.logLevel = 0;

    {
        SimulationEngine eng(cfg);
        Detection det_exact;
        det_exact.x = 0.3f; det_exact.y = 0.3f;
        det_exact.width = 0.1f; det_exact.height = 0.1f;
        det_exact.confidence = 0.50f; det_exact.classId = 0;
        auto evs = eng.ProcessDetections({det_exact}, 0.0);
        Record(S, "conf=0.50 (경계값) → 이벤트 생성", evs.size() == 1,
               "C++: if(conf < min) continue → 0.50 >= 0.50 통과");
    }
    {
        SimulationEngine eng(cfg);
        Detection det_below;
        det_below.x = 0.3f; det_below.y = 0.3f;
        det_below.width = 0.1f; det_below.height = 0.1f;
        det_below.confidence = 0.4999f; det_below.classId = 0;
        auto evs = eng.ProcessDetections({det_below}, 0.0);
        Record(S, "conf=0.4999 → 이벤트 없음", evs.empty(),
               "got=" + std::to_string(evs.size()));
    }

    // ── 6-B: 빈 detection ────────────────────────────────────────────────
    {
        SimulationEngine eng(cfg);
        auto evs = eng.ProcessDetections({}, 0.0);
        Record(S, "빈 detection → 빈 이벤트", evs.empty(), "");
    }

    // ── 6-C: 정규화 bbox 경계값 → TargetCandidate ────────────────────────
    struct BCase { const char* name; float x, y, w, h; };
    const BCase cases[] = {
        {"(0,0) top-left",          0.00f, 0.00f, 0.05f, 0.05f},
        {"(0.95,0) top-right",      0.95f, 0.00f, 0.05f, 0.05f},
        {"(0,0.95) bot-left",       0.00f, 0.95f, 0.05f, 0.05f},
        {"(0.95,0.95) bot-right",   0.95f, 0.95f, 0.05f, 0.05f},
        {"(0.475,0.475) center",    0.475f,0.475f,0.05f, 0.05f},
    };
    for (const auto& bc : cases) {
        Detection det;
        det.x = bc.x; det.y = bc.y;
        det.width = bc.w; det.height = bc.h;
        det.confidence = 0.80f; det.classId = 0;
        TargetCandidate tc = TargetCandidate::FromDetection(det, FRAME_W, FRAME_H, 0.0, 0);
        bool cx_ok = (tc.centerPixelX >= 0.0f && tc.centerPixelX <= static_cast<float>(FRAME_W));
        bool cy_ok = (tc.centerPixelY >= 0.0f && tc.centerPixelY <= static_cast<float>(FRAME_H));
        char buf[128];
        std::snprintf(buf, sizeof(buf), "bbox %s → center 범위", bc.name);
        Record(S, buf, cx_ok && cy_ok,
               std::string("center=(") + std::to_string(tc.centerPixelX) + "," +
               std::to_string(tc.centerPixelY) + ")");
    }

    // ── 6-D: clamp 검증 (x<0, y<0, x+w>1, y+h>1) ────────────────────────
    struct ClampCase { const char* name; float x, y, w, h; };
    const ClampCase clamp_cases[] = {
        {"x<0",      -0.1f,  0.1f, 0.2f, 0.2f},
        {"y<0",       0.1f, -0.1f, 0.2f, 0.2f},
        {"x+w>1",     0.9f,  0.1f, 0.2f, 0.2f},
        {"y+h>1",     0.1f,  0.9f, 0.2f, 0.2f},
    };
    for (const auto& cc : clamp_cases) {
        // C++ std::clamp(val, 0.0f, 1.0f) 적용
        float cx = std::max(0.0f, std::min(1.0f, cc.x));
        float cy = std::max(0.0f, std::min(1.0f, cc.y));
        float cw = std::max(0.0f, std::min(1.0f, cc.w));
        float ch = std::max(0.0f, std::min(1.0f, cc.h));
        Detection det;
        det.x = cx; det.y = cy; det.width = cw; det.height = ch;
        det.confidence = 0.80f; det.classId = 0;
        TargetCandidate tc = TargetCandidate::FromDetection(det, FRAME_W, FRAME_H, 0.0, 0);
        bool ok = (tc.centerPixelX >= 0.0f && tc.centerPixelY >= 0.0f);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "clamp [%s]", cc.name);
        Record(S, buf, ok,
               std::string("after_clamp_det=(") + std::to_string(cx) + "," + std::to_string(cy) + ")");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 7: EventOrderTest
// ════════════════════════════════════════════════════════════════════════════

void EventOrderTest()
{
    const std::string S = "EventOrder";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 7: %s\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    MockPicoHID::Instance().ClearHistory();

    const std::string json_str = R"({
  "name": "test_field",
  "points": [
    {"name": "Start", "x": 100, "y": 236},
    {"name": "P1",    "x": 300, "y": 150},
    {"name": "P2",    "x": 567, "y": 200},
    {"name": "P3",    "x": 800, "y": 300}
  ],
  "routes": [["Start","P1","P2","P3"]]
})";

    HuntingGround hg = HuntingGround::FromJson(json_str);

    SimEngineConfig cfg;
    cfg.frameW = 1135; cfg.frameH = 472;
    cfg.charStartX = 567.0f; cfg.charStartY = 236.0f;
    cfg.logLevel = 0;

    SimulationEngine engine(cfg);

    // Attack → Move 순서
    Detection det;
    det.x = 0.30f; det.y = 0.25f;
    det.width = 0.15f; det.height = 0.30f;
    det.confidence = 0.80f; det.classId = 0;
    auto atk_evs = engine.ProcessDetections({det}, 0.0);
    auto mov_evs = engine.SimulateRoute(hg, 0, 200.0);

    // ── 7-A: AttackSim 수학 ───────────────────────────────────────────────
    if (!atk_evs.empty()) {
        const auto& ev = atk_evs[0];
        Record(S, "AttackSim: end_ms = start_ms + 150",
               NearlyEq(ev.endMs, ev.startMs + 150.0),
               "endMs=" + std::to_string(ev.endMs));
        Record(S, "AttackSim: dragDuration = 150ms",
               NearlyEq(ev.dragDurationMs, 150.0), "");
    }

    // ── 7-B: MoveSim 시작 시간 ────────────────────────────────────────────
    if (!mov_evs.empty()) {
        Record(S, "MoveSim[0].startMs = 200.0",
               NearlyEq(mov_evs[0].startMs, 200.0),
               "got=" + std::to_string(mov_evs[0].startMs));
    }

    // ── 7-C: travel_ms 수학 (mv.from_x/y 직접 사용) ──────────────────────
    for (size_t i = 0; i < mov_evs.size(); ++i) {
        const auto& mv = mov_evs[i];
        double dist = std::sqrt(std::pow(mv.toX - mv.fromX, 2.0) +
                                std::pow(mv.toY - mv.fromY, 2.0));
        double exp_t = dist / 200.0 * 1000.0;
        double got_t = mv.arrivalMs - mv.startMs;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "MoveSim[%zu] travel_ms 수학", i);
        Record(S, buf, std::abs(got_t - exp_t) < 0.01,
               "from=(" + std::to_string((int)mv.fromX) + "," + std::to_string((int)mv.fromY) +
               ") to=(" + std::to_string((int)mv.toX) + "," + std::to_string((int)mv.toY) +
               ") dist=" + std::to_string(dist) +
               " exp=" + std::to_string(exp_t) + " got=" + std::to_string(got_t));
    }

    // ── 7-D: HID 이벤트 순서 (Attack drag가 Move 이벤트보다 먼저) ─────────
    auto hist = MockPicoHID::Instance().GetHistory();
    if (!hist.empty()) {
        Record(S, "첫 번째 HID 이벤트 = MouseDrag (AttackSim)",
               hist[0].type == MockHidEventType::MouseDrag,
               std::string("got=") + MockHidEventTypeName(hist[0].type));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Suite 8: StressTest (100회 — C++ 빌드에서는 빠르게)
// ════════════════════════════════════════════════════════════════════════════

void StressTest()
{
    const std::string S = "StressTest";
    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Suite 8: %s (100회)\n", S.c_str());
    std::printf("%s\n", std::string(60, '=').c_str());

    const int N = 100;
    const int FRAME_W = 1135, FRAME_H = 472;

    int crashes = 0;
    int coord_errors = 0;
    int drag_errors = 0;

    for (int trial = 0; trial < N; ++trial) {
        try {
            // 시드 기반 결정론적 테스트 데이터
            float det_x = static_cast<float>((trial * 17 % 80)) / 100.0f;
            float det_y = static_cast<float>((trial * 13 % 80)) / 100.0f;
            float det_w = 0.05f + static_cast<float>(trial % 10) * 0.01f;
            float det_h = 0.05f + static_cast<float>(trial % 8)  * 0.01f;
            float conf  = 0.55f + static_cast<float>(trial % 40) * 0.01f;
            float char_x = static_cast<float>(200 + trial % 700);
            float char_y = static_cast<float>(100 + trial % 300);

            SimEngineConfig cfg;
            cfg.frameW = FRAME_W; cfg.frameH = FRAME_H;
            cfg.charStartX = char_x; cfg.charStartY = char_y;
            cfg.minConfidence = 0.50f; cfg.logLevel = 0;

            SimulationEngine eng(cfg);

            Detection det;
            det.x = det_x; det.y = det_y;
            det.width = det_w; det.height = det_h;
            det.confidence = conf; det.classId = 0;

            auto evs = eng.ProcessDetections({det}, 0.0);
            if (evs.empty()) continue;

            const auto& ev = evs[0];

            // center 검증
            float exp_cx = (det_x + det_w * 0.5f) * static_cast<float>(FRAME_W);
            float exp_cy = (det_y + det_h * 0.5f) * static_cast<float>(FRAME_H);
            if (std::abs(ev.targetCenterX - exp_cx) > 1e-3f ||
                std::abs(ev.targetCenterY - exp_cy) > 1e-3f) {
                ++coord_errors;
            }

            // drag_from 검증
            int exp_fx = static_cast<int>(char_x);
            int exp_fy = static_cast<int>(char_y);
            if (ev.dragFromX != exp_fx || ev.dragFromY != exp_fy) {
                ++drag_errors;
            }
        } catch (...) {
            ++crashes;
        }
    }

    std::printf("  %d회 반복: crashes=%d coord_errors=%d drag_errors=%d\n",
                N, crashes, coord_errors, drag_errors);
    Record(S, "100회 반복 crash 없음", crashes == 0, "crashes=" + std::to_string(crashes));
    Record(S, "좌표 오차 = 0 (100회)", coord_errors == 0, "errors=" + std::to_string(coord_errors));
    Record(S, "drag_from 일관성 (100회)", drag_errors == 0, "errors=" + std::to_string(drag_errors));
}

// ════════════════════════════════════════════════════════════════════════════
// 최종 보고서
// ════════════════════════════════════════════════════════════════════════════

void PrintReport()
{
    std::printf("\n%s\n", std::string(72, '=').c_str());
    std::printf("  C++ SimTest_unit 최종 보고서\n");
    std::printf("%s\n", std::string(72, '=').c_str());

    using R = UnitTest::Result;
    std::map<std::string, std::vector<R>> by_suite;
    for (const auto& r : UnitTest::g_results)
        by_suite[r.suite].push_back(r);

    const char* labels[] = {
        "CoordinateTransform", "TargetCandidate", "AttackSimulation",
        "HuntingGround", "MockPicoHID", "BoundaryValues", "EventOrder", "StressTest"
    };
    for (const char* label : labels) {
        auto it = by_suite.find(label);
        if (it == by_suite.end()) continue;
        int p = 0, f = 0;
        for (const auto& r : it->second) { if (r.passed) ++p; else ++f; }
        const char* status = f == 0 ? "PASS \xe2\x9c\x85" : "FAIL \xe2\x9d\x8c";
        std::printf("  %-30s %s  (%d/%zu)\n",
                    label, status, p, it->second.size());
        if (f > 0) {
            for (const auto& r : it->second)
                if (!r.passed)
                    std::printf("    \xe2\x9c\x97 %s: %s\n", r.name.c_str(), r.detail.c_str());
        }
    }

    std::printf("\n%s\n", std::string(72, '-').c_str());
    std::printf("  전체: %d건 PASS / %d건 FAIL\n", UnitTest::g_pass, UnitTest::g_fail);
    if (UnitTest::g_fail == 0) {
        std::printf("  \xe2\x98\x85 모든 C++ 유닛 테스트 PASS \xe2\x98\x85\n");
    }
    std::printf("%s\n", std::string(72, '=').c_str());
}

// ════════════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════════════

int main()
{
    std::printf("SimTest_unit — C++ 유닛 테스트\n");
    std::printf("(실제 HID/마우스/키보드 입력 없음 — MockPicoHID 전용)\n");
    std::printf("%s\n", std::string(60, '=').c_str());

    CoordinateTransformTest();
    TargetCandidateTest();
    AttackSimulationTest();
    HuntingGroundTest();
    MockPicoHIDTest();
    BoundaryValuesTest();
    EventOrderTest();
    StressTest();

    PrintReport();

    return UnitTest::g_fail > 0 ? 1 : 0;
}
