"""
controller.py
-------------
피코 HID 컨트롤러.

핵심 원칙:
  - GetCursorPos 사용 안 함 (피코 HID 커서 != Windows 커서)
  - 커서 위치를 직접 추적 (self._cur_x, self._cur_y)
  - 시작 시 화면 중앙으로 커서 리셋
  - 공격 = PRESS(누름) → MOVE(드래그) → RELEASE(놓음)
  - 이동 = 현재 추적 위치 기준 dx/dy 계산 후 MOVE 1회

피코 펌웨어 프로토콜 (텍스트, 줄바꿈 종료):
  PING          → PONG 응답
  MOVE:dx:dy    → 상대 마우스 이동
  PRESS         → 마우스 왼쪽 버튼 누름
  RELEASE       → 마우스 왼쪽 버튼 놓음
  CLICK[:ms]    → PRESS + 대기 + RELEASE (ms 기본 80)
  STOP          → 버튼 강제 해제

HID 감도 스케일:
  SCALE = 0.395  (측정값: MOVE:1 → 실제 2.53px 이동)
  원하는 픽셀 P → 전송값 = P * SCALE
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
    def drag_attack(self, x: int, y: int, drag_dx: int, drag_dy: int, hold_ms: int): ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


# ------------------------------------------------------------------
# Dummy Controller (테스트용, 실제 입력 없음)
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

    def drag_attack(self, x: int, y: int, drag_dx: int = 8,
                    drag_dy: int = 0, hold_ms: int = 80):
        print(f"[DummyController] 드래그공격: ({x},{y}) drag=({drag_dx},{drag_dy})")

    # 하위호환 alias
    def click_drag(self, x: int, y: int, drag_dx: int = 8,
                   drag_dy: int = 0, hold_ms: int = 80):
        self.drag_attack(x, y, drag_dx, drag_dy, hold_ms)

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
    # 측정 결과: MOVE:1 → 실제 2.53px 이동
    # SCALE = 1 / 2.53 = 0.395
    # 즉 원하는 픽셀 P를 이동하려면 P * 0.395 를 전송
    SCALE_X = 0.395
    SCALE_Y = 0.395

    def __init__(self, port: str, baudrate: int = 115200):
        self._port      = port
        self._baudrate  = baudrate
        self._serial    = None
        self._connected = False
        self._lock      = threading.Lock()

        # 피코 커서 추적 위치 (화면 중앙에서 시작)
        self._cur_x = self.SCREEN_W // 2
        self._cur_y = self.SCREEN_H // 2

    # ── 연결 ───────────────────────────────────────────────────────
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

    def disconnect(self):
        # 버튼이 눌린 채로 끊기지 않도록 RELEASE 먼저
        try:
            self._send_text("RELEASE")
            time.sleep(0.05)
        except Exception:
            pass
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connected = False
        print("[PicoController] 연결 해제")

    # ── 커서 리셋 ──────────────────────────────────────────────────
    def _reset_cursor(self):
        """
        커서를 좌상단(0,0)으로 리셋 후 화면 중앙으로 이동 → 위치 동기화.
        MOVE:-9999:-9999 는 피코 펌웨어에서 좌상단 끝까지 이동 처리.
        """
        self._send_text("MOVE:-9999:-9999")
        time.sleep(0.6)

        # 중앙으로 이동 (스케일 적용)
        cx = self.SCREEN_W // 2   # 960
        cy = self.SCREEN_H // 2   # 540
        scaled_x = int(cx * self.SCALE_X)
        scaled_y = int(cy * self.SCALE_Y)
        self._send_text(f"MOVE:{scaled_x}:{scaled_y}")
        time.sleep(0.5)

        self._cur_x = cx
        self._cur_y = cy
        print(f"[PicoController] 커서 리셋 완료: 중앙({cx},{cy}) 전송({scaled_x},{scaled_y})")

    # ── 핵심 공격: 드래그 공격 ────────────────────────────────────
    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 8, drag_dy: int = 0,
                    hold_ms: int = 80):
        """
        몬스터 좌표(x, y)로 이동 후 PRESS → 옆으로 드래그 → RELEASE.

        흐름:
          1. 현재 추적 커서 → 목표(x, y) 로 MOVE
          2. PRESS (마우스 왼쪽 버튼 누름)
          3. drag_dx, drag_dy 만큼 드래그 MOVE
          4. RELEASE (버튼 놓음)

        drag_dx/dy: 드래그 방향 (픽셀 단위, 스케일 적용 후 전송)
                    기본 +8px 오른쪽으로 드래그 → 몬스터 클릭드래그
        hold_ms: PRESS 후 드래그 전 대기 (ms)
        """
        # ── 1. 현재 위치 → 목표로 이동 ──────────────────────────
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = int(dx * self.SCALE_X)
        sdy = int(dy * self.SCALE_Y)

        print(f"[Pico] 추적({self._cur_x},{self._cur_y}) → 목표({x},{y}) "
              f"전송MOVE({sdx},{sdy})")

        if abs(sdx) > 0 or abs(sdy) > 0:
            self._send_text(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)   # 이동 안착 대기

        # 추적 위치 업데이트
        self._cur_x = max(0, min(self.SCREEN_W, x))
        self._cur_y = max(0, min(self.SCREEN_H, y))

        # ── 2. PRESS ────────────────────────────────────────────
        self._send_text("PRESS")
        time.sleep(hold_ms / 1000.0)   # 누름 유지 (기본 80ms)

        # ── 3. 드래그 MOVE ──────────────────────────────────────
        if abs(drag_dx) > 0 or abs(drag_dy) > 0:
            drag_sdx = int(drag_dx * self.SCALE_X)
            drag_sdy = int(drag_dy * self.SCALE_Y)
            self._send_text(f"MOVE:{drag_sdx}:{drag_sdy}")
            time.sleep(0.03)

            # 드래그 후 추적 위치도 업데이트
            self._cur_x = max(0, min(self.SCREEN_W, self._cur_x + drag_dx))
            self._cur_y = max(0, min(self.SCREEN_H, self._cur_y + drag_dy))

        # ── 4. RELEASE ──────────────────────────────────────────
        self._send_text("RELEASE")
        print(f"[Pico] 드래그공격 완료 @ ({self._cur_x},{self._cur_y})")

    # ── 하위호환 alias ─────────────────────────────────────────────
    def click_drag(self, x: int, y: int,
                   drag_dx: int = 8, drag_dy: int = 0,
                   hold_ms: int = 80):
        """drag_attack 의 alias (main.py 호환용)."""
        self.drag_attack(x, y, drag_dx, drag_dy, hold_ms)

    # ── 단순 CLICK (필요 시) ───────────────────────────────────────
    def click(self, x: int, y: int, hold_ms: int = 50):
        """이동 후 단순 CLICK (드래그 없음)."""
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = int(dx * self.SCALE_X)
        sdy = int(dy * self.SCALE_Y)
        if abs(sdx) > 0 or abs(sdy) > 0:
            self._send_text(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)
        self._cur_x = max(0, min(self.SCREEN_W, x))
        self._cur_y = max(0, min(self.SCREEN_H, y))
        self._send_text(f"CLICK:{hold_ms}")

    # ── 유틸 ──────────────────────────────────────────────────────
    def stop(self):
        """버튼 강제 해제 (비상 정지용)."""
        self._send_text("RELEASE")
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
