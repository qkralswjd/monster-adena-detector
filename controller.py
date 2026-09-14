"""
controller.py
-------------
피코 HID 컨트롤러.

핵심 원칙:
  - GetCursorPos 사용 안 함 (피코 HID 커서 != Windows 커서)
  - 커서 위치를 직접 추적 (self._cur_x, self._cur_y)
  - 시작 시 화면 중앙으로 커서 리셋
  - 이동 = 현재 추적 위치 기준 dx/dy 계산 후 MOVE 1회
  - 클릭 = CLICK 1회
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional


# ------------------------------------------------------------------
# 추상 베이스 클래스
# ------------------------------------------------------------------

class BaseController(ABC):

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self): ...

    @abstractmethod
    def click_drag(self, x: int, y: int, drag_dx: int, drag_dy: int, hold_ms: int): ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


# ------------------------------------------------------------------
# Dummy Controller
# ------------------------------------------------------------------

class DummyController(BaseController):

    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        print("[DummyController] 연결 완료 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False

    def attack(self, x: int, y: int):
        print(f"[DummyController] 공격: ({x},{y})")

    def click_drag(self, x: int, y: int, drag_dx: int = 5, drag_dy: int = 0, hold_ms: int = 100):
        print(f"[DummyController] 클릭: ({x},{y})")

    def click_move(self, x: int, y: int):
        print(f"[DummyController] 이동클릭: ({x},{y})")

    @property
    def is_connected(self) -> bool:
        return self._connected


# ------------------------------------------------------------------
# Pico HID Controller
# ------------------------------------------------------------------

class PicoController(BaseController):

    # 화면 해상도
    SCREEN_W = 1920
    SCREEN_H = 1080

    # HID 감도 보정 스케일
    # 테스트 결과: MOVE:960 → 실제 1919 이동 → 비율 0.5
    # 즉 원하는 픽셀의 절반만 보내야 정확히 이동
    SCALE_X = 0.5
    SCALE_Y = 0.5

    def __init__(self, port: str, baudrate: int = 115200):
        self._port      = port
        self._baudrate  = baudrate
        self._serial    = None
        self._connected = False
        self._lock      = threading.Lock()

        # 피코 커서 추적 위치 (화면 중앙에서 시작)
        self._cur_x = self.SCREEN_W // 2
        self._cur_y = self.SCREEN_H // 2

    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baudrate, timeout=1.0)
            time.sleep(0.5)
            self._connected = True
            print(f"[PicoController] 연결 완료: {self._port} @ {self._baudrate}")

            # 커서를 화면 중앙으로 리셋
            self._reset_cursor()
            return True
        except Exception as e:
            print(f"[PicoController] 연결 실패: {e}")
            self._connected = False
            return False

    def _reset_cursor(self):
        """커서를 좌상단(0,0)으로 리셋 후 중앙으로 이동해서 위치 동기화."""
        self._send_text("MOVE:-9999:-9999")
        time.sleep(0.5)
        # 중앙으로 이동 (스케일 적용)
        cx = self.SCREEN_W // 2
        cy = self.SCREEN_H // 2
        scaled_x = int(cx * self.SCALE_X)
        scaled_y = int(cy * self.SCALE_Y)
        self._send_text(f"MOVE:{scaled_x}:{scaled_y}")
        time.sleep(0.5)
        self._cur_x = cx
        self._cur_y = cy
        print(f"[PicoController] 커서 리셋: 중앙({cx},{cy}) 전송값({scaled_x},{scaled_y})") 

    def disconnect(self):
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connected = False
        print("[PicoController] 연결 해제")

    def click_drag(self, x: int, y: int, drag_dx: int = 5,
                   drag_dy: int = 0, hold_ms: int = 80):
        """
        추적 중인 커서 위치 기준으로 목표까지 MOVE 후 CLICK.
        로그에 현재 추적 위치와 목표 위치, dx/dy 출력.
        """
        dx = x - self._cur_x
        dy = y - self._cur_y

        # 스케일 적용 (HID 감도 보정)
        sdx = int(dx * self.SCALE_X)
        sdy = int(dy * self.SCALE_Y)

        print(f"[Pico] 추적({self._cur_x},{self._cur_y}) → 목표({x},{y}) 전송MOVE({sdx},{sdy})")

        if abs(sdx) > 0 or abs(sdy) > 0:
            self._send_text(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.06)

        # 커서 추적 위치 업데이트
        self._cur_x = max(0, min(self.SCREEN_W, x))
        self._cur_y = max(0, min(self.SCREEN_H, y))

        self._send_text(f"CLICK:{hold_ms}")
        print(f"[Pico] CLICK @ ({self._cur_x},{self._cur_y})")

    def attack(self, x: int, y: int):
        self.click_drag(x, y, hold_ms=50)

    def click_move(self, x: int, y: int):
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = int(dx * self.SCALE_X)
        sdy = int(dy * self.SCALE_Y)
        self._send_text(f"MOVE:{sdx}:{sdy}")
        time.sleep(0.06)
        self._send_text("CLICK:20")
        self._cur_x = x
        self._cur_y = y

    def stop(self):
        self._send_text("STOP")

    def ping(self) -> bool:
        self._send_text("PING")
        try:
            resp = self._serial.readline().decode("utf-8", "ignore").strip()
            return resp == "PONG"
        except Exception:
            return False

    def _send_text(self, text: str):
        if not self._connected or self._serial is None:
            return
        try:
            data = (text + "\n").encode("utf-8")
            with self._lock:
                self._serial.write(data)
                self._serial.flush()
        except Exception as e:
            print(f"[PicoController] 전송 실패: {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def make_controller(cfg) -> BaseController:
    if not cfg.controller.enabled:
        c = DummyController()
        c.connect()
        return c

    if cfg.controller.type == "pico":
        c = PicoController(port=cfg.controller.port, baudrate=cfg.controller.baudrate)
        if c.connect():
            return c
        print("[Controller] Pico 연결 실패 → DummyController로 폴백")

    dummy = DummyController()
    dummy.connect()
    return dummy
