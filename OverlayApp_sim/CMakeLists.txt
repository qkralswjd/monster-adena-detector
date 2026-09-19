cmake_minimum_required(VERSION 3.20)
project(OverlayApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/external/onnxruntime)

# GPU inference: external/onnxruntime is a CPU-only ORT build (no CUDA/DML
# provider). If external/onnxruntime_dml (ORT + DirectML.dll) is present it is
# used instead, and the detector runs on the GPU via the DirectML EP (falls
# back to CPU at runtime if the EP can't be created). Configure with
# -DOVERLAY_USE_DIRECTML=OFF to force the CPU-only package.
option(OVERLAY_USE_DIRECTML "Use the DirectML-enabled ONNX Runtime package for GPU inference" ON)
set(OVERLAY_ORT_HAS_DML OFF)
if(OVERLAY_USE_DIRECTML AND EXISTS ${CMAKE_SOURCE_DIR}/external/onnxruntime_dml/lib/onnxruntime.lib)
    set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/external/onnxruntime_dml)
    set(OVERLAY_ORT_HAS_DML ON)
endif()

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${IMGUI_DIR}/backends/imgui_impl_dx11.cpp
)

set(APP_SOURCES
    src/main.cpp
    src/Window.cpp
    src/Renderer.cpp
    src/Overlay.cpp
    src/capture/ScreenCapture.cpp
    src/detection/YoloDetector.cpp
    src/detection/DummyDetector.cpp
    src/detection/DetectionWorker.cpp
    src/tracking/ObjectTracker.cpp
    src/objects/ObjectStateManager.cpp
    src/targeting/TargetManager.cpp
    src/events/DeathEventManager.cpp
    src/diagnostics/RuntimeMonitor.cpp
    src/diagnostics/CoordDiag.cpp
    src/config/AppConfig.cpp
    src/hardware/PicoHidDevice.cpp
    src/hardware/HardwareInputManager.cpp
    src/dataset/DatasetCapture.cpp
)

add_executable(OverlayApp WIN32
    ${APP_SOURCES}
    ${IMGUI_SOURCES}
)

target_include_directories(OverlayApp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(OverlayApp PRIVATE
    d3d11
    dxgi
    d3dcompiler
    dwmapi
    windowsapp
    pdh
    psapi
    hid
    setupapi
    windowscodecs
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# onnxruntime.dll is loaded at runtime and must sit next to the executable.
# models/ is copied alongside it too, so the default relative model path
# ("models/<name>.onnx") resolves regardless of the process's working directory.
add_custom_command(TARGET OverlayApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll
        $<TARGET_FILE_DIR:OverlayApp>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll
        $<TARGET_FILE_DIR:OverlayApp>
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:OverlayApp>/models
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        $<TARGET_FILE_DIR:OverlayApp>/config
)

if(OVERLAY_ORT_HAS_DML)
    add_custom_command(TARGET OverlayApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DIR}/lib/DirectML.dll
            $<TARGET_FILE_DIR:OverlayApp>
    )
    target_compile_definitions(OverlayApp PRIVATE OVERLAY_HAS_DML)
endif()

# Offline console tool: runs ONLY the Adena detector (adena_detection.* config
# keys) on a saved image and prints/saves the result. Shares YoloDetector and
# the config loader with OverlayApp; nothing else from the app.
add_executable(AdenaDetectTest
    src/tools/AdenaDetectTest.cpp
    src/detection/YoloDetector.cpp
    src/diagnostics/CoordDiag.cpp
    src/config/AppConfig.cpp
)
target_include_directories(AdenaDetectTest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ONNXRUNTIME_DIR}/include
)
target_link_libraries(AdenaDetectTest PRIVATE
    d3d11
    dxgi
    dwmapi
    windowscodecs
    ole32
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)
add_custom_command(TARGET AdenaDetectTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll
        $<TARGET_FILE_DIR:AdenaDetectTest>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll
        $<TARGET_FILE_DIR:AdenaDetectTest>
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        $<TARGET_FILE_DIR:AdenaDetectTest>/config
)
if(OVERLAY_ORT_HAS_DML)
    add_custom_command(TARGET AdenaDetectTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DIR}/lib/DirectML.dll
            $<TARGET_FILE_DIR:AdenaDetectTest>
    )
    target_compile_definitions(AdenaDetectTest PRIVATE OVERLAY_HAS_DML)
endif()
target_compile_definitions(AdenaDetectTest PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)
if(MSVC)
    target_compile_options(AdenaDetectTest PRIVATE /W4 /permissive-)
endif()

target_compile_definitions(OverlayApp PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(OverlayApp PRIVATE /W4 /permissive-)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# SimulationTest: 오프라인/시뮬레이션 콘솔 도구
#   - 실제 게임/HID 입력 없이 Detection → TargetCandidate → Attack/Move 시뮬
#   - 기존 OverlayApp 및 AdenaDetectTest 빌드에 영향 없음
#   - simulation/ 모듈은 헤더-only 설계이므로 SimulationEngine.cpp만 추가
# ─────────────────────────────────────────────────────────────────────────────
add_executable(SimulationTest
    src/tools/SimulationTest.cpp
    src/simulation/SimulationEngine.cpp
    src/detection/YoloDetector.cpp
    src/diagnostics/CoordDiag.cpp
    src/config/AppConfig.cpp
)

target_include_directories(SimulationTest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(SimulationTest PRIVATE
    d3d11
    dxgi
    dwmapi
    windowscodecs
    ole32
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# onnxruntime DLL + config/ 폴더를 출력 디렉터리에 복사
add_custom_command(TARGET SimulationTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll
        $<TARGET_FILE_DIR:SimulationTest>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll
        $<TARGET_FILE_DIR:SimulationTest>
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        $<TARGET_FILE_DIR:SimulationTest>/config
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:SimulationTest>/models
)

if(OVERLAY_ORT_HAS_DML)
    add_custom_command(TARGET SimulationTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DIR}/lib/DirectML.dll
            $<TARGET_FILE_DIR:SimulationTest>
    )
    target_compile_definitions(SimulationTest PRIVATE OVERLAY_HAS_DML)
endif()

target_compile_definitions(SimulationTest PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(SimulationTest PRIVATE /W4 /permissive-)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# SimTestUnit: C++ 독립 유닛 테스트
#   - Win32/D3D11/ONNX 의존성 없음 (헤더-only simulation 모듈만 사용)
#   - 실제 HID/마우스/키보드 입력 없음 (MockPicoHID 전용)
#   - Windows 이외 환경에서도 컴파일 가능 (std::chrono 사용)
#   - 기존 OverlayApp, AdenaDetectTest, SimulationTest 빌드에 영향 없음
# ─────────────────────────────────────────────────────────────────────────────
add_executable(SimTestUnit
    src/tools/SimTest_unit.cpp
    src/simulation/SimulationEngine.cpp   # 컴파일 단위 stub (헤더-only 분리)
)

target_include_directories(SimTestUnit PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ★ D3D11/ONNX/WIC 의존성 없음 — 순수 C++ 표준 라이브러리만 사용 ★
# target_link_libraries: 없음 (모든 구현이 헤더-only)

target_compile_definitions(SimTestUnit PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(SimTestUnit PRIVATE /W4 /permissive-)
endif()

# config/ 폴더 복사 (hunting_ground.json 참조용)
add_custom_command(TARGET SimTestUnit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        $<TARGET_FILE_DIR:SimTestUnit>/config
)
