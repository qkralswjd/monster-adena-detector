"""
detector.py
-----------
YOLOv8 기반 몬스터 탐지 모듈.
커스텀 모델 또는 기본 모델 사용 가능.
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


class YOLODetector:
    """YOLOv8 기반 탐지기."""

    def __init__(self, model_path: str = "yolov8s.pt",
                 confidence: float = 0.4,
                 iou_threshold: float = 0.45,
                 device: str = "cuda",
                 img_size: int = 640,
                 classes: Optional[List[int]] = None):
        self._conf = confidence
        self._iou = iou_threshold
        self._device = device
        self._img_size = img_size
        self._classes = classes
        self._model = None
        self._fps_ticks = []
        self._fps = 0.0
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            # GPU 워밍업
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, device=self._device)
            print(f"[Detector] YOLOv8 로드 완료: {model_path} @ {self._device}")
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
                    cls_name = self._model.names.get(cls_id, "unknown")
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
