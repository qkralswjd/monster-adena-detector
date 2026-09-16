# PROJECT_MID_AUDIT.md
> 생성일: 2026-09-16  
> 기준 커밋: `196154a` (`feat: StateMachine + WaypointMover + LootDetector 통합 구현`)  
> 브랜치: `main`  
> 미커밋(untracked): `update_config.py`

---

## 1. 전체 디렉터리 구조

```
monster-adena-detector/
├── [KEEP]         main.py                   # 메인 진입점 (StateMachine 통합)
├── [KEEP]         state_machine.py          # BotState FSM (이번 세션 신규)
├── [KEEP]         waypoint_mover.py         # 웨이포인트 이동 (이번 세션 신규)
├── [KEEP]         loot_detector.py          # HSV 아데나 탐지 (이번 세션 신규)
├── [KEEP]         level_detector.py         # 레벨/HP 인식 (OCR+HSV)
├── [KEEP]         controller.py             # PicoController / DummyController
├── [KEEP]         detector.py               # YOLODetector (best.pt)
├── [KEEP]         screen_capture.py         # mss 백그라운드 캡처
├── [KEEP]         target_selector.py        # TargetTracker (속도벡터 추적)
├── [KEEP]         overlay_window.py         # tkinter 투명 오버레이
├── [KEEP]         setup_tool.py             # GUI 설정 툴 (35KB, tkinter)
├── [KEEP]         config.json               # 전체 설정 (레퍼런스 좌표 완비)
├── [KEEP]         requirements.txt          # 의존성 (⚠ 누락 있음 → P1)
├── [KEEP]         dataset/dataset.yaml      # YOLO 데이터셋 설정
├── [KEEP]         detect_test.py            # 개발용 탐지 시각화 테스트
├── [KEEP]         test_coords.py            # 개발용 좌표 검증
│
├── [REFACTOR]     overlay_window.py         # HUD 구형 상태명 매핑 잔재 (P2)
├── [REFACTOR]     level_detector.py         # docstring "빨간색" 4곳 잔재 (P2)
├── [REFACTOR]     setup_tool.py             # 구형 config 키(pos/waypoints) 참조 (P2)
├── [REFACTOR]     main.py                   # 전체화면 2중 캡처 (P2)
│
├── [LEGACY]       test_pico.py              # 구 API(SCALE_X/SCALE_Y/_send) 기반 (P1)
├── [LEGACY]       benchmark.py             # 이중경로 버그 + 구형모델 (P3)
│
├── [DELETE CANDIDATE] fix_roi.py           # 일회성 스크립트, 이미 적용됨
├── [DELETE CANDIDATE] update_config.py     # 일회성 스크립트, 미커밋 untracked
├── [DELETE CANDIDATE] debug_hp.py          # 단순 HP HSV 샘플링 디버그용
├── [DELETE CANDIDATE] benchmark_capture.py # 하드코딩 ROI 벤치마크
├── [DELETE CANDIDATE] capture_tool.py      # tools/capture_dataset.py와 중복
├── [DELETE CANDIDATE] check_monitors.py    # 일회성 모니터 진단 스크립트
│
├── [RISK]         config.json              # model 경로 미존재 → P0
├── [RISK]         overlay_detect.py        # cap.capture() 튜플 오용 → P1
├── [RISK]         requirements.txt         # 6개 패키지 누락 → P1
├── [RISK]         test_pico.py             # 존재하지 않는 API 참조 → P1
├── [RISK]         .gitignore               # runs/, dataset/ 미제외 → P3
│
├── tools/
│   ├── [KEEP]     train.py                 # YOLOv8s 학습 (→ monster_v1)
│   ├── [KEEP]     train_v2.py              # adena 재학습 (→ monster_v2)
│   ├── [KEEP]     label_tool.py            # GrabCut 라벨링 도구
│   ├── [KEEP]     split_val.py             # train→val 20% 분할
│   ├── [KEEP]     set_roi.py               # ROI 드래그 설정
│   └── [KEEP]     capture_dataset.py       # 학습 데이터 캡처 (T키)
│
├── runs/detect/
│   ├── monster_v1/weights/best.pt          (22MB, 정상 경로)
│   ├── monster_v1-2/weights/best.pt        (22MB, 정상 경로)
│   └── runs/detect/                        ← ⚠ 이중경로 버그 (중첩 디렉터리)
│       ├── monster_v3/weights/best.pt      (22MB)
│       ├── monster_v4/weights/best.pt      (22MB)
│       └── monster_v5_nano/weights/best.pt (6MB)
│
├── dataset/images/raw/                     (1068장 .jpg, 총 875MB)
│   └── [Git 추적 중]  ← ⚠ .gitignore 미포함, repo 763MB
│
└── weights/yolo26n.pt                      (베이스 모델)
```

---

## 2. 전체 파일 수 요약

| 분류 | 수 |
|------|-----|
| 루트 Python 파일 | 25개 |
| tools/ Python 파일 | 6개 |
| 설정/문서 파일 | 3개 (config.json, requirements.txt, .gitignore) |
| dataset 이미지 | 1,068장 |
| 학습 모델(best.pt) | 5개 |
| **소스 합계** | **31개 .py** |

---

## 3. 실제 사용 중인 파일 (Runtime Import 추적)

```
main.py
  ├── screen_capture.py     (ScreenCapture)
  ├── detector.py           (YOLODetector, Detection)
  ├── target_selector.py    (TargetTracker)
  ├── controller.py         (PicoController, DummyController)
  ├── overlay_window.py     (OverlayWindow)
  └── state_machine.py      (StateMachine, BotState)
        ├── waypoint_mover.py   (WaypointMover)
        ├── loot_detector.py    (LootDetector)
        └── level_detector.py   (LevelDetector)
```

**overlay_detect.py** (별도 실행 가능, main.py와 독립):
```
overlay_detect.py
  ├── screen_capture.py
  ├── detector.py
  ├── target_selector.py
  └── overlay_window.py
```

---

## 4. 미사용 의심 파일

| 파일 | 이유 |
|------|------|
| `fix_roi.py` | 일회성 config 수정 스크립트, 이미 config.json에 반영됨 |
| `update_config.py` | 일회성 config 업데이트, 이미 config.json에 반영됨 |
| `debug_hp.py` | 단순 디버그 스크립트 (HP HSV 샘플 출력), 개발 완료 후 불필요 |
| `benchmark_capture.py` | 하드코딩된 ROI 값으로 캡처 속도만 측정 |
| `check_monitors.py` | 모니터 목록 출력 + PNG 저장, 일회성 진단 |
| `capture_tool.py` | `tools/capture_dataset.py`와 거의 동일한 기능 중복 |

---

## 5. 레거시 파일

| 파일 | 이유 |
|------|------|
| `test_pico.py` | `SCALE_X`, `SCALE_Y`, `_send()` — controller.py 구버전 기반 API. 현재 controller.py에 없음 → 실행 시 즉시 AttributeError |
| `benchmark.py` | 경로 `runs/detect/runs/detect/monster_v5_nano/...` (이중경로 버그), 현재 아키텍처와 무관 |

---

## 6. 중복 코드

| 항목 | 파일 A | 파일 B | 내용 |
|------|--------|--------|------|
| 학습 데이터 캡처 | `capture_tool.py` | `tools/capture_dataset.py` | 동일 기능 (T키로 스크린 캡처 → dataset/ 저장), 저장 경로만 약간 다름 |
| 전체화면 캡처 | `main.py`의 `capture_full_frame()` | `state_machine.py`의 `self.grab()` | **동일한 함수를 매 루프 2번 호출** (P2) |
| HP 물약 체크 | `state_machine._check_hp_and_use_potion()` | (main.py에 별도 없음, SM이 처리) | 현재는 SM에만 있어 OK |

---

## 7. 삭제 후보 목록

> **삭제 전 사용자 확인 필수** — 이번 단계에서는 삭제하지 않음

| 파일 | 삭제 이유 | 위험도 |
|------|-----------|--------|
| `fix_roi.py` | 이미 적용된 일회성 스크립트 | 낮음 |
| `update_config.py` | 이미 적용된 일회성 스크립트, untracked | 낮음 |
| `debug_hp.py` | 개발 완료 후 불필요한 디버그 스크립트 | 낮음 |
| `benchmark_capture.py` | 하드코딩 ROI, 일회성 측정 | 낮음 |
| `check_monitors.py` | 일회성 진단 도구 | 낮음 |
| `capture_tool.py` | `tools/capture_dataset.py`와 중복 | 낮음 (중복제거) |

---

## 8. 배포/실행 위험 요소 (우선순위별)

---

### [P0] 즉시 실행 장애 — main.py 시작 불가

#### P0-1: config.json의 detector.model 경로 미존재

```json
"detector": {
    "model": "runs/detect/monster_v2/weights/best.pt"
}
```

**실제 존재하는 경로:**
```
runs/detect/monster_v1/weights/best.pt          ✅
runs/detect/monster_v1-2/weights/best.pt        ✅
runs/detect/runs/detect/monster_v3/...          ✅ (이중경로)
runs/detect/runs/detect/monster_v4/...          ✅ (이중경로)
runs/detect/runs/detect/monster_v5_nano/...     ✅ (이중경로)
runs/detect/monster_v2/weights/best.pt          ❌ 존재하지 않음
```

**영향:** `main.py` 실행 시 `YOLODetector._load_model()` 에서 YOLO 로드 실패.  
detector는 None 상태로 계속 실행되나 탐지가 전혀 동작하지 않음.

**권장 수정:**
```json
"model": "runs/detect/monster_v1-2/weights/best.pt"
```
(또는 가장 최신 성능이 좋은 버전으로 교체)

---

### [P1] 배포 전 수정 권장

#### P1-1: overlay_detect.py — cap.capture() 반환값 튜플 오용

**screen_capture.py line 122:**
```python
return self._latest_frame, is_new   # (ndarray, bool) 튜플 반환
```

**overlay_detect.py line 81:**
```python
frame = cap.capture()               # frame에 (ndarray, bool) 튜플이 들어감
if frame is None:                   # 튜플은 None이 아님 → 항상 통과
    continue
detections = det.detect(frame)      # ndarray 대신 튜플 전달 → 런타임 에러
```

**영향:** `overlay_detect.py` 실행 시 즉시 `TypeError` 발생.  
(main.py는 `frame, is_new = cap.capture()` 로 올바르게 언패킹 중 → main.py는 정상)

**권장 수정:**
```python
# overlay_detect.py line 81
frame, is_new = cap.capture()
if frame is None or not is_new:
    continue
```

#### P1-2: test_pico.py — 존재하지 않는 API 참조

```python
ctrl.SCALE_X    # AttributeError — PicoController에 없음
ctrl.SCALE_Y    # AttributeError — PicoController에 없음
ctrl._send()    # AttributeError — PicoController에 없음 (_write_line이 현재 이름)
```

**영향:** `test_pico.py` 실행 시 즉시 AttributeError.  
**해결 방법:** LEGACY 표기 + 파일 상단 주석 추가, 또는 신규 API로 재작성

#### P1-3: requirements.txt — 6개 패키지 누락

**현재 내용:**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
mss>=9.0.0
```

**누락된 패키지 (코드 import 추적 결과):**
```
easyocr          # level_detector.py — 레벨 OCR 핵심
pyautogui        # main.py, state_machine.py — F키 입력
Pillow           # check_monitors.py, debug_hp.py (PIL.Image)
pyserial         # controller.py — serial.Serial (Pico UART)
keyboard         # capture_tool.py, tools/capture_dataset.py — 핫키
torch            # benchmark.py, ultralytics 내부 의존
```

**권장 추가:**
```
easyocr>=1.7.0
pyautogui>=0.9.54
Pillow>=10.0.0
pyserial>=3.5
keyboard>=0.13.5
torch>=2.0.0
```

---

### [P2] 정리/개선 필요

#### P2-1: overlay_window.py — HUD 구형 상태명 매핑 잔재

```python
# overlay_window.py line 351~354
state_clr = {
    "HUNTING":      CLR_TARGET,
    "ADENA_CHECK":  CLR_ADENA_TGT,   # ← 구형 상태명 (StateMachine에 없음)
    "DUMMY_ATTACK": CLR_DUMMY,        # ← 구형 상태명 (ATTACKING_DUMMY가 정확)
    "MOVING":       CLR_WAYPOINT,     # ← 구형 상태명 (MOVE_TO_HUNT_ZONE이 정확)
}.get(state, CLR_TEXT)
```

```python
# overlay_window.py line 248
clr = CLR_ADENA_TGT if state == "ADENA_CHECK" else CLR_ADENA   # 조건 항상 False
```

**영향:** HUD State 표시 색상이 기본값(흰색)으로만 표시됨.  
아데나 박스 색상 강조(CLR_ADENA_TGT)도 동작 안 함.  
기능은 동작하지만 시각적 피드백이 잘못됨.

**권장 수정:**
```python
state_clr = {
    "HUNTING":             CLR_TARGET,
    "LOOTING":             CLR_ADENA_TGT,
    "ATTACKING_DUMMY":     CLR_DUMMY,
    "MOVE_TO_HUNT_ZONE":   CLR_WAYPOINT,
    "IDLE":                CLR_TEXT,
    "DONE":                CLR_TEXT,
}.get(state, CLR_TEXT)
```

#### P2-2: level_detector.py — docstring "빨간색" 4곳 잔재

```python
# line 70: "# 빨간색 픽셀 마스크 (두 HSV 범위 OR)"
# line 184: hp_color: str = "red",  # 하위 호환용 (무시됨, 항상 빨간색)
# line 201: print("[LevelDetector] 초기화 완료 (HP: 빨간색 HSV, 전체 열 합계 방식)")
# line 301~305: """HP ROI에서 빨간색 픽셀 열 비율로 HP% 계산.
#   - HSV 빨간색 범위 (H=0~10, H=170~180)"""
```

**영향:** 코드는 정상 동작 (실제 로직은 파란색 HSV=100~140), 혼동 유발.

#### P2-3: main.py — 전체화면 2중 캡처 (매 루프)

```python
# main.py 루프 내부
sm.update(frame, detections)          # 내부에서 self.grab() 호출 → 전체화면 1회
# ...
if sm.state not in (BotState.IDLE, BotState.DONE):
    _full = capture_full_frame()       # 동일한 capture_full_frame → 또 1회
    hp_pct    = sm.level_det.read_hp(_full)
    level_now = sm.level_det.read_level(_full)
```

**영향:** 매 루프마다 전체화면 캡처를 2회 수행. CPU/메모리 낭비.  
level_det.read_hp/read_level는 내부적으로 0.5s/2s 캐시가 있어 실제 OCR은 스킵되지만, mss 캡처 자체는 2번 실행됨.

**권장 수정:** `sm.update()`가 캡처한 frame을 반환하거나, 오버레이용 HP/Level 값을 SM에서 노출.

#### P2-4: setup_tool.py — 신규 config 키 미반영

```python
# setup_tool.py line 96~100
dp  = cfg.get("dummy", {}).get("pos")          # 구형 단순클릭 좌표 (미사용)
wps = cfg.get("movement", {}).get("waypoints", [])  # 구형 단일 waypoints (빈 리스트)
```

**실제 config:**
```json
"dummy.pos": null,
"movement.waypoints": [],          ← 빈 배열 (구형)
"movement.hunt_waypoints": [...],  ← 실제 사용 (setup_tool에서 미지원)
"movement.patrol_waypoints": [...] ← 실제 사용 (setup_tool에서 미지원)
```

**영향:** setup_tool.py에서 저장한 허수아비/웨이포인트 좌표가 현재 StateMachine에서 읽히지 않음.  
(setup_tool의 설정 저장 기능이 실질적으로 무효화됨)

---

### [P3] 단순 코드 정리

#### P3-1: .gitignore — dataset/, runs/ 미포함

```
# 현재 .gitignore에 없음:
dataset/
runs/
```

**현황:**
- dataset/: 1,068장 이미지, 875MB — **git 추적 중**
- runs/: 학습 모델 파일 5개(best.pt), 201MB — **git 추적 중**
- git pack 크기: **763MB** (비정상적으로 큰 repo)

**영향:** `git push` 속도 극히 느림. GitHub 100MB 단일파일 제한 위반 가능성.  
(현재 push는 sandbox 환경이라 통과했을 수 있음)

#### P3-2: runs/ 이중경로 버그

```
runs/detect/monster_v3/   ← 존재하지 않음 (정상경로)
runs/detect/runs/detect/monster_v3/   ← 실제 저장된 경로 (이중경로)
```

`tools/train_v2.py` 또는 YOLO가 `/monster-adena-detector` 디렉터리에서 실행되지 않고, `runs/detect/` 내부에서 실행되어 발생한 것으로 추정. 경로 혼동 주의.

#### P3-3: benchmark.py 이중경로

```python
model = YOLO('runs/detect/runs/detect/monster_v5_nano/weights/best.pt')
```

우연히 이중경로가 맞아서 파일은 존재하지만, 코드 의도와 실제 경로가 불일치.

---

## 9. DB/Schema 불일치

**해당 없음.** 이 프로젝트는 SQLite/DB를 사용하지 않음.

---

## 10. 외부 API / Provider 관련 코드

| 외부 시스템 | 현재 상태 |
|------------|-----------|
| **Raspberry Pi Pico** (UART HID) | 활성 사용 — `controller.py` PicoController |
| **YOLO ultralytics** (추론) | 활성 사용 — `detector.py` |
| **easyocr** (레벨 OCR) | 활성 사용 — `level_detector.py` |
| **mss** (스크린 캡처) | 활성 사용 — `screen_capture.py` |
| **pyautogui** (F키 입력) | 활성 사용 — `main.py`, `state_machine.py` |

---

## 11. 보안 위험

| 항목 | 내용 | 위험도 |
|------|------|--------|
| 직렬 포트 하드코딩 | `config.json: "port": "COM4"` | 낮음 (개발용, 공개 시 무관) |
| 모델 파일 git 추적 | `*.pt` 주석처리로 best.pt 추적 허용 | 낮음 (저작권 이슈 있을 수 있음) |
| 게임 자동화 | 리니지 클래식 게임 매크로 → 게임사 정책 위반 가능 | 프로젝트 본질적 위험 |

---

## 12. 성능 위험

| 항목 | 내용 | 영향 |
|------|------|------|
| 전체화면 2중 캡처 | 매 루프 `capture_full_frame()` 2회 | CPU 낭비 |
| easyocr 초기화 | 첫 `read_level()` 호출 시 수초 대기 | 시작 시 딜레이 |
| LevelDetector 캐시 간격 | 레벨: 2초, HP: 0.5초 | 적정 (문제없음) |
| YOLO 추론 | `device: "cuda"` — GPU 없으면 CPU fallback 매우 느림 | CUDA 환경 필수 |

---

## 13. 빌드 위험

| 항목 | 내용 |
|------|------|
| Python 구문 검사 | main.py, state_machine.py, waypoint_mover.py, loot_detector.py — ✅ 모두 통과 |
| ctypes.windll | `controller.py`, `overlay_window.py`, `test_coords.py` — **Windows 전용** (Linux/Mac 실행 불가) |
| `keyboard` 패키지 | Linux에서 `sudo` 없이 실행 불가 |
| `mss.MSS()` vs `mss.mss()` | `check_monitors.py`는 대문자 `MSS()`, 나머지는 소문자 `mss()` — mss 버전 따라 호환성 확인 필요 |

---

## 14. 우선순위 정리

### P0 — 즉시 수정 (실행 불가)
- [x] ~~분석 완료~~ → **미수정**
- [ ] `config.json` detector.model 경로를 실존하는 경로로 수정
  - 권장: `"runs/detect/monster_v1-2/weights/best.pt"` 또는 monster_v5_nano

### P1 — 배포 전 수정
- [ ] `overlay_detect.py` cap.capture() 튜플 언패킹 수정
- [ ] `test_pico.py` 구 API 제거 또는 LEGACY 주석 추가
- [ ] `requirements.txt` 누락 패키지 6개 추가

### P2 — 개선 권장
- [ ] `overlay_window.py` HUD state_clr 딕셔너리를 BotState.name 기준으로 업데이트
- [ ] `overlay_window.py` line 248 adena 상태 조건 LOOTING으로 수정
- [ ] `level_detector.py` "빨간색" docstring 4곳 "파란색"으로 수정
- [ ] `main.py` 전체화면 2중 캡처 제거 (SM이 반환하거나 공유)
- [ ] `setup_tool.py` drag_from/hunt_waypoints/patrol_waypoints 지원 추가

### P3 — 코드 정리
- [ ] `.gitignore` 에 `dataset/`, `runs/` 추가
- [ ] 삭제 후보 6개 파일 정리 (사용자 확인 후)
- [ ] `benchmark.py` 경로 수정 또는 삭제
- [ ] `update_config.py` .gitignore에 추가 또는 삭제

---

## 15. 다음 작업 순서 제안

```
Step 1 (P0 즉시):
  config.json의 detector.model 경로 수정
  → "runs/detect/monster_v1-2/weights/best.pt" 로 변경

Step 2 (P1 한 번에):
  requirements.txt 누락 패키지 6개 추가
  overlay_detect.py cap.capture() → frame, is_new = cap.capture() 수정
  test_pico.py 상단에 LEGACY 주석 추가 (또는 신규 API로 재작성)

Step 3 (P2 UI/시각화):
  overlay_window.py HUD state_clr 딕셔너리 신규 상태명으로 업데이트
  level_detector.py "빨간색" docstring 4곳 수정

Step 4 (P2 성능):
  main.py 전체화면 2중 캡처 제거

Step 5 (P2 기능):
  setup_tool.py 신규 config 키 지원 추가
  (drag_from/drag_to, hunt_waypoints, patrol_waypoints)

Step 6 (P3 정리):
  .gitignore 업데이트 (dataset/, runs/ 추가)
  삭제 후보 파일 제거 (사용자 확인 후)

Step 7 (기능 확장):
  실제 Windows 환경에서 런타임 테스트
  monster_v2 모델 학습 완료 후 config.json model 경로 최신화
```

---

## 16. 변경된 파일 목록 (이번 세션 기준)

| 파일 | 상태 | 커밋 |
|------|------|------|
| `config.json` | 수정됨 | `196154a` |
| `main.py` | 수정됨 | `196154a` |
| `state_machine.py` | 신규 생성 | `196154a` |
| `waypoint_mover.py` | 신규 생성 | `196154a` |
| `loot_detector.py` | 신규 생성 | `196154a` |
| `update_config.py` | 신규 생성 (untracked) | 미커밋 |
| `level_detector.py` | 이전 세션 수정 | `ade014d` |
| `debug_ocr.py` | 이전 세션 수정 | `ade014d` |

---

*감사(Audit) 완료. 이 보고서는 읽기 전용 분석 결과이며 코드 수정은 포함하지 않습니다.*  
*2단계 작업(실제 수정)은 사용자 승인 후 진행합니다.*
