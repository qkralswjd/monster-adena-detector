"""
screen_capture.py
-----------------
mss 기반 고속 화면 캡처 모듈.
ROI 지정 및 전체 화면 캡처 지원.
"""

import cv2
import numpy as np
import mss
import time
from typing import Optional, Tuple


class ScreenCapture:
    def __init__(self, monitor: int = 1):
        self._sct = mss.mss()
        self._monitor_idx = monitor
        self._monitor = self._sct.monitors[monitor]
        self._roi: Optional[dict] = None
        self._fps_ticks = []
        self._fps = 0.0

    def set_roi(self, x: int, y: int, w: int, h: int):
        self._roi = {"top": y, "left": x, "width": w, "height": h}
        print(f"[Capture] ROI 설정: ({x},{y}) {w}x{h}")

    def clear_roi(self):
        self._roi = None

    def capture(self) -> Optional[np.ndarray]:
        region = self._roi if self._roi else self._monitor
        try:
            shot = self._sct.grab(region)
            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self._tick_fps()
            return frame
        except Exception as e:
            print(f"[Capture] 캡처 실패: {e}")
            return None

    def capture_full(self) -> Optional[np.ndarray]:
        """ROI 무시하고 전체 화면 캡처."""
        try:
            shot = self._sct.grab(self._monitor)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            return None

    def select_roi_interactive(self, frame: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """마우스 드래그로 ROI 선택."""
        clone = frame.copy()
        roi_data = {"start": None, "end": None, "done": False}

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_data["start"] = (x, y)
                roi_data["end"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and roi_data["start"]:
                roi_data["end"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                roi_data["end"] = (x, y)
                roi_data["done"] = True

        win = "ROI 선택 (드래그 후 Enter)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, mouse_cb)

        while True:
            disp = clone.copy()
            if roi_data["start"] and roi_data["end"]:
                cv2.rectangle(disp, roi_data["start"], roi_data["end"], (0,255,0), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32) and roi_data["done"]:
                break
            if key == 27:
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        s, e = roi_data["start"], roi_data["end"]
        x = min(s[0], e[0]); y = min(s[1], e[1])
        w = abs(e[0]-s[0]); h = abs(e[1]-s[1])
        if w > 10 and h > 10:
            ox = self._monitor["left"]
            oy = self._monitor["top"]
            return (x + ox, y + oy, w, h)
        return None

    def _tick_fps(self):
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            e = self._fps_ticks[-1] - self._fps_ticks[0]
            if e > 0:
                self._fps = round((len(self._fps_ticks)-1) / e, 1)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._monitor["width"], self._monitor["height"]
