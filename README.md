# monster_tracker

게임 화면에서 **몬스터를 클릭으로 지정 → 사망하면 자동으로 다음 몬스터 선택** 하는 추적 시스템.

> ⚠️ ByteTrack / 칼만필터 제거 — 죽고 새로 스폰되는 구조에 ID 추적은 의미 없음.  
> **IoU 매칭 + 사망 감지 + nearest 자동 선택** 으로 재설계.

---

## 구조

```
화면 캡처 (mss)
    ↓
YOLOv8s 탐지  (GPU)
    ↓
타겟 갱신
  ├── 클릭 → 해당 위치 몬스터 지정
  ├── 매 프레임 IoU 매칭 → 동일 몬스터 추적
  └── 소실/사망 감지 → 자동으로 nearest 선택
    ↓
시각화 (OpenCV)
```

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 메인 루프, 마우스 클릭 처리, 키 입력 |
| `detector.py` | YOLOv8s GPU 탐지, Detection 데이터클래스 |
| `target_selector.py` | nearest / click 타겟 선택, IoU 매칭 |
| `death_detector.py` | 타겟 소실 타이머 → 사망 확정 판정 |
| `screen_capture.py` | mss 화면 캡처, ROI 선택 UI |
| `visualizer.py` | 박스, 코너 강조, HUD 그리기 |
| `config.json` | 전체 설정 |

---

## 설치

```bash
pip install -r requirements.txt

# GPU PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 실행

```bash
python main.py
```

## 조작

| 입력 | 동작 |
|------|------|
| **마우스 클릭** | 해당 몬스터를 타겟으로 지정 |
| `C` | 타겟 해제 |
| `R` | ROI 재설정 |
| `Q` / `ESC` | 종료 |

---

## 설정 (`config.json`)

```jsonc
{
  "detector": {
    "model": "yolov8s.pt",   // 커스텀 .pt 경로 가능
    "confidence": 0.4,
    "device": "cuda"
  },
  "target": {
    "death_timeout_sec": 1.5, // 소실 몇 초 후 사망 확정
    "death_iou_thresh": 0.3,  // IoU 이 미만이면 소실로 판정
    "click_radius": 60        // 클릭 인식 반경 (픽셀)
  }
}
```

---

## 동작 원리

### 타겟 추적 (매 프레임)
```
현재 타겟 bbox ↔ 새 탐지 bbox  →  IoU 계산
  IoU ≥ 0.3  → 같은 몬스터 → bbox 업데이트
  IoU < 0.3  → 소실 타이머 시작
    1.5초 지속 → 사망 확정 → nearest 자동 선택
```

### 자동 선택
- 타겟 없을 때 → 화면 중앙에서 가장 가까운 몬스터 자동 지정
- 클릭으로 지정 후 `C` 해제하기 전까지는 자동 선택 OFF
