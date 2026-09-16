"""
screen_capture.py
-----------------
mss 로 모니터 화면을 BGR numpy 배열로 캡처.
백그라운드 스레드가 계속 캡처 → 메인루프는 최신 프레임을 즉시 가져감.
캡처와 추론이 병렬로 동작해 FPS 향상.

사용:
  cap = ScreenCapture(monitor=1, roi=cfg.get("roi"))
  frame = cap.capture()   # numpy BGR (H, W, 3) - 항상 최신 프레임
  print(cap.width, cap.height)
"""

import time
import threading
import numpy as np

try:
    import mss
except ImportError:
    raise ImportError("mss 설치 필요: pip install mss")


class ScreenCapture:

    def __init__(self, monitor: int = 1, roi: dict = None):
        """
        monitor: mss monitors 인덱스.
        roi: {"enabled": true, "x": 300, "y": 50, "width": 1300, "height": 850}
        """
        self._mon_idx = monitor

        # 좌표 계산용으로만 mss 한 번 사용
        with mss.mss() as sct:
            mon = sct.monitors[monitor]
            mon_left = mon["left"]
            mon_top  = mon["top"]

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
            self._capture_region = {
                "left":   mon_left,
                "top":    mon_top,
                "width":  1920,
                "height": 1080,
            }
            self._roi_x = 0
            self._roi_y = 0
            self.width  = 1920
            self.height = 1080

        self.left = mon_left + self._roi_x
        self.top  = mon_top  + self._roi_y

        # 백그라운드 캡처용
        self._lock         = threading.Lock()
        self._latest_frame = None
        self._frame_id     = 0   # 새 프레임마다 증가
        self._last_read_id = -1  # 마지막으로 읽은 프레임 ID
        self._running      = False
        self._thread       = None

        self._fps_ticks = []
        self._fps       = 0.0

        print(f"[Capture] monitors[{monitor}] "
              f"left={self.left} top={self.top} "
              f"{self.width}x{self.height}")
        print(f"[Capture] 캡처 좌표 (0,0) = 화면 ({self.left},{self.top})")

        # 백그라운드 캡처 스레드 시작
        self._start_bg()

    def _start_bg(self):
        """백그라운드 캡처 스레드 시작."""
        self._running = True
        self._thread  = threading.Thread(target=self._bg_loop, daemon=True)
        self._thread.start()

    def _bg_loop(self):
        """백그라운드에서 계속 캡처. 메인루프보다 빠르게 돌며 최신 프레임 유지."""
        with mss.mss() as sct:
            while self._running:
                try:
                    raw   = sct.grab(self._capture_region)
                    frame = np.frombuffer(raw.rgb, dtype=np.uint8)
                    frame = frame.reshape((raw.height, raw.width, 3))
                    # RGB → BGR 변환 (mss.rgb는 RGB 순서)
                    frame = frame[:, :, ::-1].copy()
                    with self._lock:
                        self._latest_frame = frame
                        self._frame_id    += 1
                    self._tick_fps()
                except Exception as e:
                    print(f"[Capture] 캡처 실패: {e}")
                    time.sleep(0.01)

    def capture(self):
        """
        최신 캡처 프레임 반환 (블로킹 없음).
        반환: (frame, is_new)
          - frame  : BGR numpy (H, W, 3) 또는 None
          - is_new : 이전 호출 이후 새 프레임이면 True
        """
        with self._lock:
            is_new = (self._frame_id != self._last_read_id)
            if is_new:
                self._last_read_id = self._frame_id
            return self._latest_frame, is_new

    def stop(self):
        """백그라운드 캡처 스레드 종료."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 60:
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
    time.sleep(0.5)  # 백그라운드 스레드 안정화
    print(f"모니터 크기: {cap.width}x{cap.height}")
    print(f"모니터 위치: left={cap.left}, top={cap.top}")

    frame = cap.capture()
    if frame is not None:
        print(f"캡처 성공: shape={frame.shape}")
    else:
        print("캡처 실패")

    time.sleep(1.0)
    print(f"캡처 FPS: {cap.fps}")
    cap.stop()
