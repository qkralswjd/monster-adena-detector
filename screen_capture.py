"""
screen_capture.py
-----------------
mss 로 모니터 전체 화면을 1920x1080 BGR numpy 배열로 캡처.

좌표 원칙:
  캡처 이미지 (0,0) = 모니터 좌상단 (0,0)
  캡처 이미지 (cx, cy) = 화면 절대좌표 (cx, cy)
  → 좌표 변환 없음.

사용:
  cap = ScreenCapture(monitor=1)
  frame = cap.capture()   # numpy BGR (H, W, 3)
  print(cap.width, cap.height)
"""

import time
import numpy as np

try:
    import mss
except ImportError:
    raise ImportError("mss 설치 필요: pip install mss")


class ScreenCapture:

    def __init__(self, monitor: int = 1, roi: dict = None):
        """
        monitor: mss monitors 인덱스.
          monitors[0] = 전체 가상 화면
          monitors[1] = 첫 번째 물리 모니터 (보통 1920x1080)
        roi: {"enabled": true, "x": 300, "y": 50, "width": 1300, "height": 850}
          enabled=True 면 해당 영역만 캡처 → FPS 향상
          enabled=False 면 전체 화면 캡처
        """
        self._sct     = mss.mss()
        self._mon_idx = monitor
        self._monitor = self._sct.monitors[monitor]

        mon_left = self._monitor["left"]
        mon_top  = self._monitor["top"]

        # ROI 적용 여부
        if roi and roi.get("enabled", False):
            self._capture_region = {
                "left":   mon_left + roi["x"],
                "top":    mon_top  + roi["y"],
                "width":  roi["width"],
                "height": roi["height"],
            }
            self._roi_x = roi["x"]
            self._roi_y = roi["y"]
            self.width  = roi["width"]
            self.height = roi["height"]
            print(f"[Capture] ROI 모드: x={roi['x']} y={roi['y']} "
                  f"{roi['width']}x{roi['height']}")
        else:
            self._capture_region = self._monitor
            self._roi_x = 0
            self._roi_y = 0
            self.width  = self._monitor["width"]
            self.height = self._monitor["height"]

        self.left = mon_left + self._roi_x
        self.top  = mon_top  + self._roi_y

        self._fps_ticks = []
        self._fps = 0.0

        print(f"[Capture] monitors[{monitor}] "
              f"left={self.left} top={self.top} "
              f"{self.width}x{self.height}")
        print(f"[Capture] 캡처 좌표 (0,0) = 화면 ({self.left},{self.top})")

    def capture(self) -> np.ndarray:
        """
        화면 캡처 (전체 or ROI).
        반환: BGR numpy (H, W, 3)
        실패 시: None
        """
        try:
            raw = self._sct.grab(self._capture_region)
            frame = np.array(raw)           # BGRA
            frame = frame[:, :, :3]         # BGR
            self._tick_fps()
            return frame
        except Exception as e:
            print(f"[Capture] 캡처 실패: {e}")
            return None

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            elapsed = self._fps_ticks[-1] - self._fps_ticks[0]
            if elapsed > 0:
                self._fps = round((len(self._fps_ticks) - 1) / elapsed, 1)

    @property
    def fps(self) -> float:
        return self._fps


# ── 단독 실행: 캡처 확인 ──────────────────────────────────
if __name__ == "__main__":
    cap = ScreenCapture(monitor=1)
    print(f"모니터 크기: {cap.width}x{cap.height}")
    print(f"모니터 위치: left={cap.left}, top={cap.top}")

    frame = cap.capture()
    if frame is not None:
        print(f"캡처 성공: shape={frame.shape}")
        print(f"좌상단 픽셀 BGR: {frame[0][0]}")
        print(f"중앙 픽셀 BGR:   {frame[cap.height//2][cap.width//2]}")
    else:
        print("캡처 실패")
