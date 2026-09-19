// SimulationTest.cpp
// ==================
// 오프라인 시뮬레이션 콘솔 도구.
// 실제 게임/HID 입력 없이 다음을 테스트합니다:
//
//   1. 저장된 이미지에서 Monster Detection (YoloDetector)
//   2. Detection → TargetCandidate 변환
//   3. 목표 선택 (SimulationEngine)
//   4. 공격 이벤트 생성 + MockPicoHID 로그
//   5. HuntingGround JSON 로드 + 웨이포인트 이동 시뮬레이션
//   6. 세션 결과 JSON 저장
//
// Usage:
//   SimulationTest [--image <path>] [--ground <json>]
//                  [--model <onnx>] [--classes <txt>] [--conf <0..1>]
//                  [--config <ini>] [--out <result.json>] [--log <0|1|2>]
//
// Exit codes: 0=OK, 1=arg/file error, 2=model error

#include "simulation/SimulationEngine.h"

#include "config/AppConfig.h"
#include "detection/YoloDetector.h"

#include <Windows.h>
#include <d3d11.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

using Microsoft::WRL::ComPtr;

// ── 내부 유틸 ────────────────────────────────────────────────────────────────
namespace {

std::string WideToUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, w.data(),
        static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
        s.data(), size, nullptr, nullptr);
    return s;
}

// WIC 이미지 로드 (AdenaDetectTest.cpp 동일 패턴)
struct RawImage {
    UINT width = 0, height = 0;
    std::vector<uint8_t> bgra;
};

bool LoadRawImage(IWICImagingFactory* wic, const std::wstring& path, RawImage& out) {
    ComPtr<IWICBitmapDecoder> dec;
    if (FAILED(wic->CreateDecoderFromFilename(path.c_str(), nullptr,
            GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)))
        return false;
    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return false;
    ComPtr<IWICFormatConverter> conv;
    if (FAILED(wic->CreateFormatConverter(&conv))) return false;
    if (FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
            WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)))
        return false;
    UINT w = 0, h = 0;
    conv->GetSize(&w, &h);
    if (!w || !h) return false;
    out.width = w; out.height = h;
    out.bgra.resize(static_cast<size_t>(w) * h * 4);
    return SUCCEEDED(conv->CopyPixels(nullptr, w * 4,
        static_cast<UINT>(out.bgra.size()), out.bgra.data()));
}

// 결과 JSON 파일 저장
bool SaveJson(const std::string& path, const std::string& json) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << json;
    return true;
}

void PrintUsage() {
    std::printf(
        "\n"
        "SimulationTest - 오프라인 시뮬레이션 테스트 도구\n"
        "Usage:\n"
        "  SimulationTest [--image <path.png>] [--ground <hunting_ground.json>]\n"
        "                 [--model <monster.onnx>] [--classes <classes.txt>]\n"
        "                 [--conf <0..1>] [--config <app_config.ini>]\n"
        "                 [--out <result.json>] [--log <0|1|2>]\n"
        "\n"
        "Options:\n"
        "  --image    테스트할 PNG/JPG 이미지 (생략 시 합성 Detection 사용)\n"
        "  --ground   HuntingGround JSON 파일 (생략 시 내장 예시 사용)\n"
        "  --model    Monster ONNX 모델 경로 (기본: config에서 읽음)\n"
        "  --classes  클래스명 파일 경로 (기본: config에서 읽음)\n"
        "  --conf     최소 confidence 임계값 (기본: 0.50)\n"
        "  --config   app_config.ini 경로 (기본: config/app_config.ini)\n"
        "  --out      결과 JSON 출력 경로 (기본: sim_result.json)\n"
        "  --log      로그 수준 0=quiet 1=summary 2=verbose (기본: 2)\n"
        "\n"
        "Exit: 0=OK, 1=arg/file error, 2=model error\n"
    );
}

// ── 허수아비 합성 Detection (이미지 없이 테스트용) ──────────────────────────
std::vector<Detection> MakeDummyDetections() {
    // ROI 1135×472 기준, 정규화 [0,1] 좌표
    Detection d1;
    d1.classId   = 0; d1.className = "monster";
    d1.confidence = 0.82f;
    d1.x = 0.30f; d1.y = 0.25f; d1.width = 0.15f; d1.height = 0.30f;

    Detection d2;
    d2.classId   = 0; d2.className = "monster";
    d2.confidence = 0.61f;
    d2.x = 0.65f; d2.y = 0.40f; d2.width = 0.12f; d2.height = 0.28f;

    Detection d3;
    d3.classId   = 1; d3.className = "adena";
    d3.confidence = 0.74f;
    d3.x = 0.50f; d3.y = 0.60f; d3.width = 0.05f; d3.height = 0.04f;

    return { d1, d2, d3 };
}

// ── 내장 예시 HuntingGround ──────────────────────────────────────────────────
HuntingGround MakeDefaultHuntingGround() {
    HuntingGround hg;
    hg.name = "test_field";

    // 사냥터 좌표 (ROI 1135×472 기준 픽셀)
    hg.AddPoint("Start",  100.0f, 236.0f);
    hg.AddPoint("P1",     300.0f, 150.0f);
    hg.AddPoint("P2",     567.0f, 200.0f);
    hg.AddPoint("P3",     800.0f, 300.0f);
    hg.AddPoint("P4",     950.0f, 180.0f);
    hg.AddPoint("Return", 100.0f, 236.0f);

    // 루트 A: 직진 순환
    hg.AddRoute({ "Start", "P1", "P2", "P3", "P4", "Return" });
    // 루트 B: 역방향
    hg.AddRoute({ "Return", "P4", "P3", "P2", "P1", "Start" });

    return hg;
}

} // namespace

// ── wmain ────────────────────────────────────────────────────────────────────
int wmain(int argc, wchar_t** argv) {
    SetConsoleOutputCP(CP_UTF8);

    // ── 인수 파싱 ────────────────────────────────────────────────────────────
    std::wstring imagePath, groundPath, outPath;
    std::string  configPath  = "config/app_config.ini";
    std::string  modelOverride, classesOverride;
    float        confOverride = 0.50f;
    int          logLevel     = 2;
    std::string  outPathStr   = "sim_result.json";

    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        auto next = [&]() -> std::wstring {
            return (i + 1 < argc) ? std::wstring(argv[++i]) : std::wstring{};
        };
        auto nextStr = [&]() -> std::string { return WideToUtf8(next()); };

        if      (a == L"--image")   imagePath     = next();
        else if (a == L"--ground")  groundPath    = next();
        else if (a == L"--model")   modelOverride = nextStr();
        else if (a == L"--classes") classesOverride = nextStr();
        else if (a == L"--conf")    confOverride  = static_cast<float>(_wtof(next().c_str()));
        else if (a == L"--config")  configPath    = nextStr();
        else if (a == L"--out")     outPathStr    = nextStr();
        else if (a == L"--log")     logLevel      = _wtoi(next().c_str());
        else if (a == L"-h" || a == L"--help") { PrintUsage(); return 0; }
        else { std::printf("Unknown argument: %s\n", WideToUtf8(a).c_str()); PrintUsage(); return 1; }
    }

    std::printf("\n======================================================\n");
    std::printf("  SimulationTest  (오프라인 시뮬레이션 - 입력 없음)\n");
    std::printf("======================================================\n\n");

    // ── 설정 로드 ────────────────────────────────────────────────────────────
    AppConfig appCfg = LoadAppConfig(configPath);
    // detection 필드가 Monster 모델 경로 (config 기본값 기준)
    DetectionAppConfig monCfg = appCfg.detection;
    if (!modelOverride.empty())   monCfg.modelPath      = modelOverride;
    if (!classesOverride.empty()) monCfg.classNamesPath = classesOverride;

    if (logLevel >= 1) {
        std::printf("[Config] config    = %s\n", configPath.c_str());
        std::printf("[Config] model     = %s\n", monCfg.modelPath.c_str());
        std::printf("[Config] classes   = %s\n", monCfg.classNamesPath.c_str());
        std::printf("[Config] conf      = %.2f\n", confOverride);
        std::printf("[Config] out       = %s\n\n", outPathStr.c_str());
    }

    // ── SimulationEngine 초기화 ──────────────────────────────────────────────
    SimEngineConfig simCfg;
    simCfg.frameW        = 1135;
    simCfg.frameH        = 472;
    simCfg.minConfidence = confOverride;
    simCfg.logLevel      = logLevel;
    SimulationEngine engine(simCfg);

    // ──────────────────────────────────────────────────────────────────────────
    // STEP 1: Monster Detection
    // ──────────────────────────────────────────────────────────────────────────
    std::printf("--- STEP 1: Monster Detection ---\n");

    std::vector<Detection> detections;

    if (!imagePath.empty()) {
        // 실제 이미지 → YoloDetector
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ComPtr<IWICImagingFactory> wic;
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)))) {
            std::printf("ERROR: WIC 팩토리 생성 실패\n");
            return 1;
        }

        RawImage img;
        if (!LoadRawImage(wic.Get(), imagePath, img)) {
            std::printf("ERROR: 이미지 로드 실패: %s\n", WideToUtf8(imagePath).c_str());
            return 1;
        }
        std::printf("Image: %s  (%u x %u)\n", WideToUtf8(imagePath).c_str(), img.width, img.height);

        YoloDetector detector;
        detector.SetClassNames(LoadClassNames(monCfg.classNamesPath));
        if (!detector.Initialize(monCfg.modelPath)) {
            std::printf("ERROR: 모델 로드 실패: %s\n  %s\n",
                monCfg.modelPath.c_str(), detector.GetLastError().c_str());
            return 2;
        }
        detector.SetConfidenceThreshold(confOverride);
        std::printf("Model loaded: %s\n", monCfg.modelPath.c_str());

        // D3D11 디바이스 (텍스처 업로드용)
        ComPtr<ID3D11Device>        device;
        ComPtr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL level{};
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
            &device, &level, &context);
        if (FAILED(hr))
            hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
                &device, &level, &context);
        if (FAILED(hr)) { std::printf("ERROR: D3D11 디바이스 생성 실패\n"); return 1; }

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width            = img.width;
        desc.Height           = img.height;
        desc.MipLevels        = 1;
        desc.ArraySize        = 1;
        desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage            = D3D11_USAGE_DEFAULT;
        D3D11_SUBRESOURCE_DATA init{};
        init.pSysMem     = img.bgra.data();
        init.SysMemPitch = img.width * 4;

        ComPtr<ID3D11Texture2D> tex;
        if (FAILED(device->CreateTexture2D(&desc, &init, &tex))) {
            std::printf("ERROR: D3D11 텍스처 생성 실패\n"); return 1;
        }

        detections = detector.Detect(device.Get(), context.Get(), tex.Get());
        std::printf("Detection: %zu 개  (inference %.1f ms)\n\n",
            detections.size(), detector.GetLastInferenceMs());

        for (size_t i = 0; i < detections.size(); i++) {
            const auto& d = detections[i];
            std::printf("  [%zu] class=%s  conf=%.3f  bbox(x=%.3f y=%.3f w=%.3f h=%.3f)\n",
                i, d.className.c_str(), d.confidence, d.x, d.y, d.width, d.height);
        }

    } else {
        // 이미지 없음 → 허수아비 합성 Detection
        std::printf("(이미지 미지정 - 허수아비 합성 Detection 사용)\n");
        detections = MakeDummyDetections();
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& d = detections[i];
            std::printf("  [%zu] class=%s  conf=%.3f  bbox(x=%.3f y=%.3f w=%.3f h=%.3f)\n",
                i, d.className.c_str(), d.confidence, d.x, d.y, d.width, d.height);
        }
        std::printf("\n");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // STEP 2: TargetCandidate 변환 + 목표 선택 + 공격 시뮬
    // ──────────────────────────────────────────────────────────────────────────
    std::printf("--- STEP 2: TargetCandidate + 공격 시뮬레이션 ---\n");
    auto attackEvents = engine.ProcessDetections(detections);

    if (attackEvents.empty()) {
        std::printf("  (공격 이벤트 없음 - confidence 기준 미달 또는 detection 없음)\n");
    } else {
        for (const auto& ev : attackEvents) {
            std::printf("  [AttackSim] target#%d  center(%.1f, %.1f)  conf=%.3f\n",
                ev.targetId, ev.targetCenterX, ev.targetCenterY, ev.confidence);
            std::printf("             drag (%d,%d)→(%d,%d)  %.0f ms\n",
                ev.dragFromX, ev.dragFromY, ev.dragToX, ev.dragToY, ev.dragDurationMs);
        }
    }
    std::printf("\n");

    // ──────────────────────────────────────────────────────────────────────────
    // STEP 3: HuntingGround 로드 + 이동 시뮬레이션
    // ──────────────────────────────────────────────────────────────────────────
    std::printf("--- STEP 3: HuntingGround + 이동 시뮬레이션 ---\n");

    HuntingGround hg;
    if (!groundPath.empty()) {
        try {
            hg = HuntingGround::LoadFromFile(WideToUtf8(groundPath));
            std::printf("HuntingGround: '%s'  points=%zu  routes=%zu\n",
                hg.name.c_str(), hg.points.size(), hg.routes.size());
        } catch (const std::exception& e) {
            std::printf("WARNING: HuntingGround 로드 실패 (%s) - 내장 예시 사용\n", e.what());
            hg = MakeDefaultHuntingGround();
        }
    } else {
        std::printf("(--ground 미지정 - 내장 예시 HuntingGround 사용)\n");
        hg = MakeDefaultHuntingGround();
    }

    // 웨이포인트 목록 출력
    if (logLevel >= 1) {
        std::printf("  Points:\n");
        for (const auto& wp : hg.points)
            std::printf("    %-12s (%.0f, %.0f)\n", wp.name.c_str(), wp.x, wp.y);
        std::printf("  Routes: %zu 개\n\n", hg.routes.size());
    }

    // 루트 0 이동 시뮬
    std::printf("--- STEP 3-1: Route[0] 이동 시뮬레이션 ---\n");
    auto moveEvents = engine.SimulateRoute(hg, 0);
    if (moveEvents.empty()) {
        std::printf("  (이동 이벤트 없음)\n");
    } else {
        for (const auto& mv : moveEvents) {
            std::printf("  [MoveSim] %s → %s  (%.0f,%.0f)→(%.0f,%.0f)  %.0f ms\n",
                mv.fromPoint.c_str(), mv.toPoint.c_str(),
                mv.fromX, mv.fromY, mv.toX, mv.toY,
                mv.arrivalMs - mv.startMs);
        }
    }
    std::printf("\n");

    // ──────────────────────────────────────────────────────────────────────────
    // STEP 4: MockPicoHID 전체 이벤트 로그 출력
    // ──────────────────────────────────────────────────────────────────────────
    std::printf("--- STEP 4: MockPicoHID 이벤트 로그 ---\n");
    const auto& hidHistory = MockPicoHID::Instance().GetHistory();
    std::printf("  총 %zu 개 이벤트 기록됨 (실제 HID 전달 없음)\n", hidHistory.size());
    if (logLevel >= 2) {
        for (size_t i = 0; i < hidHistory.size(); i++) {
            const auto& e = hidHistory[i];
            std::printf("  [%02zu] %-12s  t=%7.1fms  (%4d,%4d)→(%4d,%4d)  tag=%s\n",
                i, MockHidEventTypeName(e.type), e.timestampMs,
                e.x, e.y, e.toX, e.toY, e.tag.c_str());
        }
    }
    std::printf("\n");

    // ──────────────────────────────────────────────────────────────────────────
    // STEP 5: 세션 결과 JSON 저장
    // ──────────────────────────────────────────────────────────────────────────
    std::printf("--- STEP 5: 결과 JSON 저장 ---\n");
    SimSessionResult result = engine.FinalizeSession();
    std::string json = result.ToJson();

    if (SaveJson(outPathStr, json)) {
        std::printf("  결과 저장: %s  (%zu bytes)\n", outPathStr.c_str(), json.size());
    } else {
        std::printf("  WARNING: 결과 저장 실패: %s\n", outPathStr.c_str());
    }

    // ── HuntingGround JSON 저장 (예시 생성 시에만) ──────────────────────────
    if (groundPath.empty()) {
        std::string hgJsonPath = "config/hunting_ground.json";
        if (hg.SaveToFile(hgJsonPath))
            std::printf("  HuntingGround 예시 저장: %s\n", hgJsonPath.c_str());
    }

    // ── 요약 출력 ────────────────────────────────────────────────────────────
    std::printf("\n======================================================\n");
    std::printf("  시뮬레이션 완료 요약\n");
    std::printf("======================================================\n");
    std::printf("  총 프레임:       %d\n", result.totalFrames);
    std::printf("  Detection 프레임: %d\n", result.framesWithDet);
    std::printf("  후보 합계:        %d\n", result.candidatesTotal);
    std::printf("  공격 이벤트:      %d\n", result.attacksSimulated);
    std::printf("  이동 이벤트:      %d\n", result.movesSimulated);
    std::printf("  HID 로그 수:     %zu\n", result.hidEvents.size());
    std::printf("  결과 파일:        %s\n", outPathStr.c_str());
    std::printf("  실제 HID 전달:    없음 (오프라인 시뮬레이션)\n");
    std::printf("======================================================\n\n");

    return 0;
}
