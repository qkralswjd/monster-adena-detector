# monster_tracker

게임 화면에서 **단일 타겟 몬스터를 클릭으로 지정**하고, ID를 유지하며 고정 추적하는 시스템.

---

## 아키텍처

```
Screen Capture (mss)
      │
      ▼
YOLOv8s Detector  ──── GPU (RTX 2060 최적)
      │
      ▼
ByteTracker (다중 ID 관리)
  ├── Kalman Filter  (8차원 상태벡터: cx,cy,w,h,vx,vy,vw,vh)
  └── High/Low Conf 2단계 매칭
      │
      ▼
TargetTracker (단일 타겟 고정 추적)
  ├── Re-identification  (거리 + 크기 + 속도방향)
  ├── Motion Prediction  (칼만 예측으로 화면 소실 보완)
  └── ConfidenceManager  (boost on detect / decay on miss)
      │
      ▼
Visualizer + DataLogger
```

---

## 모듈 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 메인 루프, 클릭 타겟 지정, 키 입력 처리 |
| `screen_capture.py` | mss 화면 캡처 + ROI 선택 UI |
| `detector.py` | YOLOv8s GPU 탐지, Detection 데이터클래스 |
| `kalman_filter.py` | 8차원 칼만필터 (predict / update / gating_distance) |
| `byte_tracker.py` | ByteTrack 구현, High/Low conf 2단계 매칭 |
| `confidence_manager.py` | detection 시 boost, miss 시 decay, lost 확정 |
| `target_tracker.py` | 단일 타겟 고정 추적 + Re-ID + 칼만 예측 |
| `data_logger.py` | CSV 로깅, 경로 포인트 반환 |
| `visualizer.py` | 트랙 박스, 경로, HUD, 스냅샷 표시 |
| `config.json` | 전체 설정 (detector / tracker / target / logger) |

---

## 요구사항

- Python 3.9+
- CUDA 지원 GPU (RTX 2060 기준 YOLOv8s 권장)
- Windows (mss 화면 캡처)

### 설치

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. (선택) GPU 버전 PyTorch 별도 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. YOLOv8s 모델 다운로드 (첫 실행 시 자동)
# 또는 커스텀 모델: config.json > detector.model 경로 지정
```

### 의존성 목록

```
ultralytics>=8.0.0   # YOLOv8
opencv-python>=4.8.0
numpy>=1.24.0
mss>=9.0.0           # 화면 캡처
scipy>=1.11.0        # 헝가리안 매칭
lap>=0.4.0           # Linear Assignment Problem
```

---

## 실행

```bash
cd monster_tracker
python main.py
```

### 조작키

| 키 | 동작 |
|----|------|
| **마우스 클릭** | 해당 위치의 몬스터를 타겟으로 지정 |
| `ESC` / `Q` | 종료 |
| `C` | 타겟 해제 |
| `S` | 로그 저장 |
| `R` | ROI 재설정 |

---

## 설정 (`config.json`)

```jsonc
{
  "detector": {
    "model": "yolov8s.pt",      // 모델 경로 (커스텀 .pt 가능)
    "confidence": 0.4,          // 탐지 최소 confidence
    "device": "cuda"            // "cuda" or "cpu"
  },
  "tracker": {
    "track_high_thresh": 0.5,   // High-conf 매칭 임계값
    "track_low_thresh": 0.1,    // Low-conf 매칭 임계값
    "track_buffer": 30,         // 트랙 유지 버퍼 프레임
    "max_lost_frames": 60       // 최대 소실 허용 프레임
  },
  "target": {
    "reid_dist_thresh": 150,    // Re-ID 거리 임계값 (픽셀)
    "lost_confirm_sec": 3.0,    // 소실 확정까지 대기 시간
    "prediction_max_frames": 30 // 칼만 예측 최대 프레임
  }
}
```

---

## 커스텀 모델 학습 (선택)

게임 몬스터 전용 모델로 정확도를 높이려면:

```bash
# 1. 데이터셋 준비 (YOLO 포맷)
# 2. 학습
yolo train model=yolov8s.pt data=monster_dataset.yaml epochs=100 imgsz=640

# 3. config.json 업데이트
# "model": "runs/detect/train/weights/best.pt"
```

---

## 알고리즘

### ByteTrack
- **1단계**: High-conf detection (≥0.5) × 기존 트랙 IoU 매칭
- **2단계**: Low-conf detection (0.1~0.5) × 미매칭 트랙 IoU 매칭
- 사라진 트랙을 버퍼에 유지하다가 재등장 시 동일 ID 복구

### Kalman Filter (8차원)
- 상태벡터: `[cx, cy, w, h, vx, vy, vw, vh]`
- 화면에서 몬스터 소실 시 예측 위치 계산
- Mahalanobis 거리로 gating (비현실적 매칭 제거)

### Re-identification
- 거리 점수 + 크기 점수 + 속도방향 점수 조합
- 칼만 예측 위치와 비교하여 동일 타겟 복구

### Confidence Manager
- 탐지 성공: `confidence += boost (0.1)`
- 탐지 실패: `confidence -= decay (0.05)`
- `lost_confirm_sec` 초 이상 낮으면 타겟 소실 확정
