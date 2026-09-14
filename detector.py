"""
detector.py
-----------
YOLOv8 커스텀 모델 전용 탐지 모듈.

학습된 모델: runs/detect/monster_v1/weights/best.pt
  class_id=0 → monster
  class_id=1 → adena

반드시 커스텀 best.pt 만 사용한다.
COCO 사전학습 클래스는 절대 탐지하지 않는다.
탐지 결과를 Detection 객체 리스트로 반환.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class Detection:
    """탐지 결과 하나."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    class_id: int = 0
    class_name: str = "monster"

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def bbox(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def tlbr(self):
        """top-left bottom-right 형식."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def area(self) -> int:
        return self.w * self.h


# 커스텀 모델 클래스 정의 (best.pt 학습 기준 고정)
CLASS_MONSTER = 0   # monster
CLASS_ADENA   = 1   # adena
KNOWN_CLASSES = {CLASS_MONSTER: "monster", CLASS_ADENA: "adena"}


class YOLODetector:
    """커스텀 best.pt 전용 탐지기 (monster / adena 2클래스)."""

    def __init__(self, model_path: str = "runs/detect/monster_v1/weights/best.pt",
                 confidence: float = 0.4,
                 iou_threshold: float = 0.45,
                 device: str = "cuda",
                 img_size: int = 640,
                 classes: Optional[List[int]] = None):
        self._conf = confidence
        self._iou = iou_threshold
        self._device = device
        self._img_size = img_size
        # classes 파라미터는 무시하고 항상 커스텀 클래스만 사용
        # (best.pt 자체가 2클래스만 학습되어 있어 자동 제한되지만 명시)
        self._classes = None   # best.pt 는 전 클래스 = monster+adena 뿐
        self._model = None
        self._model_class_names: dict = {}
        self._fps_ticks = []
        self._fps = 0.0
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model_class_names = self._model.names  # {0:'monster', 1:'adena'}

            # 로드된 클래스 목록 출력
            print(f"[Detector] 커스텀 모델 로드: {model_path} @ {self._device}")
            print(f"[Detector] 탐지 클래스: "
                  + ", ".join(f"{k}={v}" for k, v in self._model_class_names.items()))

            # 예상 클래스 검증 (monster=0, adena=1)
            if self._model_class_names.get(0) != "monster":
                print(f"[Detector] ⚠ class_id=0 이 'monster' 가 아님: "
                      f"{self._model_class_names.get(0)} → 확인 필요")
            if self._model_class_names.get(1) != "adena":
                print(f"[Detector] ⚠ class_id=1 이 'adena' 가 아님: "
                      f"{self._model_class_names.get(1)} → 확인 필요")

            # GPU 워밍업
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, device=self._device)
            print(f"[Detector] 워밍업 완료 - 준비됨")
        except Exception as e:
            print(f"[Detector] 모델 로드 실패: {e}")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None or frame is None:
            return []
        try:
            results = self._model(
                frame,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._img_size,
                device=self._device,
                classes=self._classes,
                verbose=False
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # 커스텀 모델 클래스만 허용 (0=monster, 1=adena)
                    # best.pt 외 다른 모델이 실수로 로드되더라도 필터링
                    if cls_id not in KNOWN_CLASSES:
                        continue

                    cls_name = self._model_class_names.get(cls_id, KNOWN_CLASSES.get(cls_id, "unknown"))
                    detections.append(Detection(
                        x=x1, y=y1,
                        w=x2-x1, h=y2-y1,
                        confidence=round(conf, 3),
                        class_id=cls_id,
                        class_name=cls_name
                    ))
            self._tick_fps()
            return detections
        except Exception as e:
            print(f"[Detector] 탐지 실패: {e}")
            return []

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            e = self._fps_ticks[-1] - self._fps_ticks[0]
            if e > 0:
                self._fps = round((len(self._fps_ticks)-1)/e, 1)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
