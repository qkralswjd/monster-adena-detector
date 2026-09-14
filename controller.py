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
  원하는 픽셀 P → 전송값 = round(P * SCALE)

좌표 일치 조건 (중요!):
  mss 캡처 해상도 == 게임 렌더링 해상도 == SCREEN_W x SCREEN_H
  세 값이 모두 같아야 탐지 좌표 = 피코 이동 목표 좌표가 정확히 일치.
  Windows DPI 스케일(125%/150%) 사용 시 mss 캡처 크기가 달라지므로
  반드시 실제 mss monitors[idx] 크기를 확인 후 SCREEN_W/H 설정할 것.
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

    def click(self, x: int, y: int, hold_ms: int = 50):
        print(f"[DummyController] 클릭: ({x},{y})")

    def stop(self):
        pass

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

    # HID 감도 보정 스케일
    # 측정 결과: MOVE:1 → 실제 2.53px 이동
    # SCALE = 1 / 2.53 = 0.395
    SCALE_X = 0.395
    SCALE_Y = 0.395

    def __init__(self, port: str, baudrate: int = 115200,
                 screen_w: int = 2560, screen_h: int = 1920):
        """
        Args:
            port      : 시리얼 포트 (예: "COM4")
            baudrate  : 통신 속도
            screen_w  : 게임 화면 너비 (mss 캡처 너비와 반드시 동일해야 함)
            screen_h  : 게임 화면 높이 (mss 캡처 높이와 반드시 동일해야 함)

        중요: screen_w / screen_h 는 mss monitors[idx].width / height 와 같아야
              탐지 좌표 (0~screen_w) 와 피코 이동 목표가 정확히 일치합니다.
        """
        self._port      = port
        self._baudrate  = baudrate
        self._serial    = None
        self._connected = False
        self._lock      = threading.Lock()

        # 게임 화면 해상도 (mss 캡처 크기와 동일해야 함)
        self.SCREEN_W = screen_w
        self.SCREEN_H = screen_h

        # 피코 커서 추적 위치 (게임 화면 중앙에서 시작)
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
            print(f"[PicoController] 게임 해상도: {self.SCREEN_W}x{self.SCREEN_H} "
                  f"(mss 캡처 크기와 반드시 일치해야 좌표 정확)")
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
        커서를 물리적 좌상단(0,0)으로 이동 후 게임 화면 중앙으로 이동.

        ⚠ 전제 조건:
          MOVE:-9999:-9999 → HID 커서가 물리적 좌상단(0,0)으로 이동
          이후 MOVE:scaled_cx:scaled_cy → 게임 모니터 중앙

        ⚠ 좌표 정확도 조건:
          self.SCREEN_W/H == mss monitors[capture_idx].width/height
          이 둘이 같아야 "프레임 내 cx/cy" == "피코 이동 목표" 가 일치.
        """
        self._send_text("MOVE:-9999:-9999")
        time.sleep(0.6)

        # 게임 화면 중앙으로 이동
        cx = self.SCREEN_W // 2
        cy = self.SCREEN_H // 2
        scaled_x = round(cx * self.SCALE_X)
        scaled_y = round(cy * self.SCALE_Y)
        self._send_text(f"MOVE:{scaled_x}:{scaled_y}")
        time.sleep(0.5)

        self._cur_x = cx
        self._cur_y = cy
        print(f"[PicoController] 커서 리셋: 게임 중앙({cx},{cy}) → 전송({scaled_x},{scaled_y})")

    # ── 핵심 공격: 드래그 공격 ────────────────────────────────────
    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 8, drag_dy: int = 0,
                    hold_ms: int = 80):
        """
        몬스터 좌표(x, y)로 이동 후 PRESS → 드래그 → RELEASE.

        x, y : 프레임 내 절대 좌표 (0~SCREEN_W, 0~SCREEN_H)
               = mss 캡처 픽셀 좌표 = YOLO 탐지 결과 좌표
               피코 리셋이 (0,0)→중앙 이므로 그대로 사용 가능

        ⚠ x,y 가 정확하려면 SCREEN_W/H == mss 캡처 크기 이어야 함.
        """
        # ── 1. 현재 추적 위치 → 목표로 이동 ────────────────────
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = round(dx * self.SCALE_X)
        sdy = round(dy * self.SCALE_Y)

        print(f"[Pico] 추적({self._cur_x},{self._cur_y}) → 목표({x},{y}) "
              f"전송MOVE({sdx},{sdy})")

        if abs(sdx) > 0 or abs(sdy) > 0:
            self._send_text(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)

        # 추적 위치 업데이트
        self._cur_x = max(0, min(self.SCREEN_W, x))
        self._cur_y = max(0, min(self.SCREEN_H, y))

        # ── 2. PRESS ────────────────────────────────────────────
        self._send_text("PRESS")
        time.sleep(hold_ms / 1000.0)

        # ── 3. 드래그 MOVE ──────────────────────────────────────
        if abs(drag_dx) > 0 or abs(drag_dy) > 0:
            drag_sdx = round(drag_dx * self.SCALE_X)
            drag_sdy = round(drag_dy * self.SCALE_Y)
            self._send_text(f"MOVE:{drag_sdx}:{drag_sdy}")
            time.sleep(0.03)
            self._cur_x = max(0, min(self.SCREEN_W, self._cur_x + drag_dx))
            self._cur_y = max(0, min(self.SCREEN_H, self._cur_y + drag_dy))

        # ── 4. RELEASE ──────────────────────────────────────────
        self._send_text("RELEASE")
        print(f"[Pico] 드래그공격 완료 @ 추적({self._cur_x},{self._cur_y})")

    # ── 단순 CLICK ────────────────────────────────────────────────
    def click(self, x: int, y: int, hold_ms: int = 50):
        """이동 후 단순 CLICK (드래그 없음). 아데나 줍기용."""
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = round(dx * self.SCALE_X)
        sdy = round(dy * self.SCALE_Y)
        if abs(sdx) > 0 or abs(sdy) > 0:
            self._send_text(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)
        self._cur_x = max(0, min(self.SCREEN_W, x))
        self._cur_y = max(0, min(self.SCREEN_H, y))
        self._send_text(f"CLICK:{hold_ms}")
        print(f"[Pico] 클릭: 목표({x},{y}) 전송MOVE({sdx},{sdy})")

    # ── 하위호환 alias ─────────────────────────────────────────────
    def click_drag(self, x: int, y: int,
                   drag_dx: int = 8, drag_dy: int = 0,
                   hold_ms: int = 80):
        self.drag_attack(x, y, drag_dx, drag_dy, hold_ms)

    # ── 유틸 ──────────────────────────────────────────────────────
    def stop(self):
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
