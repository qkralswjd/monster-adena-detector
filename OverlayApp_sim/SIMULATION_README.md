# Simulation Module — 오프라인/시뮬레이션 테스트 문서

> **기존 OverlayApp, AdenaDetectTest 빌드에 영향 없음.**  
> 실제 마우스/키보드/HID 장치에 절대 전달하지 않음.

---

## 1. 새로 추가된 파일 목록

### 핵심 헤더 (`src/simulation/`)

| 파일 | 역할 |
|------|------|
| `MockPicoHID.h` | 실제 HID 대신 이벤트 큐+히스토리에만 기록하는 Mock HID Singleton |
| `TargetCandidate.h` | `Detection`/`TrackedObject` → `TargetCandidate` 변환 구조체, `AttackSimEvent`, `MoveSimEvent` |
| `HuntingGround.h` | 웨이포인트/루트 JSON 저장·로드 (외부 라이브러리 없음) |
| `SimulationEngine.h` | 시뮬레이션 파이프라인 핵심 클래스 (헤더-only) |
| `SimulationEngine.cpp` | 컴파일 단위 stub (추후 구현 분리 시 사용) |

### 콘솔 도구 (`src/tools/`)

| 파일 | 역할 |
|------|------|
| `SimulationTest.cpp` | `SimulationTest.exe` 진입점, 5단계 테스트 파이프라인 |

### 설정/배치

| 파일 | 역할 |
|------|------|
| `config/hunting_ground.json` | 사냥터 좌표 예시 (`test_field`, 6개 웨이포인트, 2개 루트) |
| `build_simulation.bat` | CMake + MSVC 빌드 배치파일 |

### 수정된 기존 파일

| 파일 | 변경 내용 |
|------|-----------|
| `CMakeLists.txt` | `SimulationTest` 타겟 추가 (기존 타겟 변경 없음) |

---

## 2. CMake 변경사항

```cmake
# 추가된 내용 (기존 타겟 수정 없음)
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
    d3d11 dxgi dwmapi windowscodecs ole32
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# POST_BUILD: onnxruntime.dll, config/, models/ 복사
```

**빌드 타겟 목록 (전체):**
| 타겟 | 상태 |
|------|------|
| `OverlayApp` | 기존 유지 (변경 없음) |
| `AdenaDetectTest` | 기존 유지 (변경 없음) |
| `SimulationTest` | **신규 추가** |

---

## 3. 빌드 방법

### 방법 A: 배치 파일 (권장)

```bat
REM Release 빌드
build_simulation.bat

REM Debug 빌드
build_simulation.bat Debug

REM 빌드 폴더 삭제
build_simulation.bat clean
```

### 방법 B: CMake 직접

```bat
REM OverlayApp 폴더에서 실행
cmake -B build -S . -A x64 -DOVERLAY_USE_DIRECTML=ON
cmake --build build --config Release --target SimulationTest -j4
```

### 방법 C: 전체 빌드 (OverlayApp + AdenaDetectTest + SimulationTest)

```bat
cmake -B build -S . -A x64
cmake --build build --config Release
```

**결과 경로:** `build\Release\SimulationTest.exe`

---

## 4. 테스트 방법

### 4-1. 기본 실행 (허수아비 합성 Detection)

이미지 없이 합성 Detection 3개로 파이프라인 전체 테스트:

```bat
build\Release\SimulationTest.exe --log 2
```

출력 예:
```
======================================================
  SimulationTest  (오프라인 시뮬레이션 - 입력 없음)
======================================================

--- STEP 1: Monster Detection ---
(이미지 미지정 - 허수아비 합성 Detection 사용)
  [0] class=monster  conf=0.820  bbox(x=0.300 y=0.250 w=0.150 h=0.300)
  [1] class=monster  conf=0.610  bbox(x=0.650 y=0.400 w=0.120 h=0.280)
  [2] class=adena    conf=0.740  bbox(x=0.500 y=0.600 w=0.050 h=0.040)

--- STEP 2: TargetCandidate + 공격 시뮬레이션 ---
  [MockHID] MouseDrag  t=1.2ms  (567,236)→(425,295)  tag=AttackSim:target_0
  [AttackSim] target#0  center(425.3, 294.8)  conf=0.820
             drag (567,236)→(425,294)  150 ms

--- STEP 3: HuntingGround + 이동 시뮬레이션 ---
(--ground 미지정 - 내장 예시 HuntingGround 사용)
  Points:
    Start        (100, 236)
    P1           (300, 150)
    ...

--- STEP 4: MockPicoHID 이벤트 로그 ---
  총 7 개 이벤트 기록됨 (실제 HID 전달 없음)
  [ 0] MouseDrag     t=    1.2ms  ( 567, 236)→( 425, 294)  tag=AttackSim:target_0
  [ 1] MouseMove     t=    2.1ms  ( 425, 236)→(   0,   0)  tag=MoveSim:from_Start
  ...

--- STEP 5: 결과 JSON 저장 ---
  결과 저장: sim_result.json  (3847 bytes)
```

### 4-2. 실제 이미지로 Monster Detection 테스트

```bat
build\Release\SimulationTest.exe ^
    --image dataset\sample_roi.png ^
    --model models\monster_best.onnx ^
    --classes config\classes_monster.txt ^
    --conf 0.50 ^
    --out sim_result.json ^
    --log 2
```

### 4-3. HuntingGround JSON 파일 지정

```bat
build\Release\SimulationTest.exe ^
    --ground config\hunting_ground.json ^
    --log 1
```

### 4-4. 조용한 모드 (결과 파일만)

```bat
build\Release\SimulationTest.exe --log 0 --out output\result.json
```

---

## 5. 생성되는 JSON 로그 예시

`sim_result.json` 구조:

```json
{
  "summary": {
    "totalFrames": 1,
    "framesWithDet": 1,
    "candidatesTotal": 2,
    "attacksSimulated": 1,
    "movesSimulated": 5
  },
  "attackEvents": [
    {
      "event": "AttackSim",
      "targetId": 0,
      "targetCenterX": 425.25,
      "targetCenterY": 294.80,
      "confidence": 0.8200,
      "startMs": 1.20,
      "endMs": 151.20,
      "dragFromX": 567,
      "dragFromY": 236,
      "dragToX": 425,
      "dragToY": 294,
      "dragDurationMs": 150.00
    }
  ],
  "moveEvents": [
    {
      "event": "MoveSim",
      "fromPoint": "Start",
      "toPoint": "P1",
      "fromX": 100.00,
      "fromY": 236.00,
      "toX": 300.00,
      "toY": 150.00,
      "startMs": 2.50,
      "arrivalMs": 1082.50,
      "arrived": true
    },
    {
      "event": "MoveSim",
      "fromPoint": "P1",
      "toPoint": "P2",
      "fromX": 300.00,
      "fromY": 150.00,
      "toX": 567.00,
      "toY": 200.00,
      "startMs": 1082.50,
      "arrivalMs": 2454.36,
      "arrived": true
    }
  ],
  "hidLog": [
    {
      "type": "MouseDrag",
      "timestampMs": 1.20,
      "x": 567, "y": 236,
      "toX": 425, "toY": 294,
      "button": 0, "keyCode": 0,
      "durationMs": 150.00,
      "tag": "AttackSim:target_0"
    },
    {
      "type": "MouseMove",
      "timestampMs": 2.50,
      "x": 425, "y": 236,
      "toX": 0, "toY": 0,
      "button": 0, "keyCode": 0,
      "durationMs": 0.00,
      "tag": "MoveSim:from_Start"
    },
    {
      "type": "Delay",
      "timestampMs": 2.55,
      "x": 0, "y": 0,
      "toX": 0, "toY": 0,
      "button": 0, "keyCode": 0,
      "durationMs": 1080.00,
      "tag": "MoveSim:travel_Start_to_P1"
    },
    {
      "type": "MouseMove",
      "timestampMs": 2.60,
      "x": 300, "y": 150,
      "toX": 0, "toY": 0,
      "button": 0, "keyCode": 0,
      "durationMs": 0.00,
      "tag": "MoveSim:arrive_P1"
    }
  ]
}
```

---

## 6. 모듈 아키텍처

```
Detection (YoloDetector)
         │
         ▼ FromDetection()
  TargetCandidate            ◄── TrackedObject::FromTrackedObject()
         │
         ▼ SelectTarget() (center 우선)
  SimulationEngine
    ├── MakeAttackEvent()
    │       └── AttackSimEvent  ──► MockPicoHID::Drag()
    └── SimulateRoute()
            └── MoveSimEvent   ──► MockPicoHID::Move() / Delay()
                                         │
                                         ▼
                              이벤트 큐 + 히스토리 저장
                              (실제 HID/마우스/키보드 없음)
                                         │
                                         ▼
                              SimSessionResult::ToJson()
```

---

## 7. 기존 프로젝트 보호 확인

| 항목 | 상태 |
|------|------|
| `OverlayApp` 빌드 | ✅ 변경 없음 |
| `AdenaDetectTest` 빌드 | ✅ 변경 없음 |
| `YoloDetector` 수정 | ✅ 없음 (읽기 전용 사용) |
| `Detection.h` 수정 | ✅ 없음 |
| `TrackedObject.h` 수정 | ✅ 없음 |
| `PicoHidDevice.h` 수정 | ✅ 없음 |
| `HardwareInputManager.h` 수정 | ✅ 없음 |
| 기존 dataset/ 변경 | ✅ 없음 |
| 기존 models/ 변경 | ✅ 없음 |
| 실제 HID 전달 | ✅ 없음 (MockPicoHID만 사용) |

---

## 8. hunting_ground.json 포맷

```json
{
  "name": "필드명",
  "points": [
    {"name": "P1", "x": 100, "y": 200},
    {"name": "P2", "x": 300, "y": 200}
  ],
  "routes": [
    ["P1", "P2"],
    ["P2", "P1"]
  ]
}
```

- `x`, `y`: ROI 기준 픽셀 좌표 (ROI = 1135×472)
- `routes`: 웨이포인트 이름 순서 목록, 여러 루트 지원
- `SimulationEngine::SimulateRoute(hg, routeIndex)` 로 실행
