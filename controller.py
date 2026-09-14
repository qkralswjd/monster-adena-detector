"""
controller.py
-------------
피코 HID 컨트롤러.

동작 원리:
  1. 연결 시 MOVE:-9999:-9999 → 피코 커서를 전체 좌상단(0,0)으로 리셋
  2. 게임이 가장 왼쪽 모니터(left=-1920)에 있으면
     피코 (0,0) = 게임 모니터 좌상단 → 프레임 좌표 = 피코 이동 좌표 일치
  3. YOLO 탐지 cx=824 → _cur_x 기준 dx 계산 → MOVE 전송

피코 펌웨어 프로토콜 (텍스트, 줄바꿈 종료):
  PING          → PONG 응답
  MOVE:dx:dy    → 상대 마우스 이동
  PRESS         → 마우스 왼쪽 버튼 누름
  RELEASE       → 마우스 왼쪽 버튼 놓음
  CLICK[:ms]    → PRESS + 대기 + RELEASE
  STOP          → 버튼 강제 해제

HID 감도 스케일:
  SCALE = 0.395  (측정값: MOVE:1 → 실제 2.53px 이동)
  보내고 싶은 픽셀 P → 전송값 = round(P * SCALE)
"""

import time
import threading
from abc import ABC, abstractmethod


# ------------------------------------------------------------------
# 추상 베이스
# ------------------------------------------------------------------
class BaseController(ABC):
    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def disconnect(self): ...
    @abstractmethod
    def drag_attack(self, x, y, drag_dx, drag_dy, hold_ms): ...
    @abstractmethod
    def click(self, x, y, hold_ms): ...
    @abstractmethod
    def stop(self): ...
    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


# ------------------------------------------------------------------
# Dummy (테스트용)
# ------------------------------------------------------------------
class DummyController(BaseController):
    def __init__(self):
        self._connected = False

    def connect(self):
        self._connected = True
        print("[DummyController] 연결 완료 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False

    def drag_attack(self, x, y, drag_dx=8, drag_dy=0, hold_ms=80):
        print(f"[DummyController] 드래그공격: ({x},{y}) drag=({drag_dx},{drag_dy})")

    def attack_click(self, x, y, hold_ms=80):
        print(f"[DummyController] 공격클릭: ({x},{y}) hold={hold_ms}ms")

    def click(self, x, y, hold_ms=50):
        print(f"[DummyController] 클릭: ({x},{y})")

    def stop(self):
        pass

    def click_drag(self, x, y, drag_dx=8, drag_dy=0, hold_ms=80):
        self.drag_attack(x, y, drag_dx, drag_dy, hold_ms)

    @property
    def is_connected(self):
        return self._connected


# ------------------------------------------------------------------
# Pico HID Controller
# ------------------------------------------------------------------
class PicoController(BaseController):

    # MOVE:1 → 실제 2.53px 이동 → SCALE = 1/2.53 = 0.395
    SCALE_X = 0.395
    SCALE_Y = 0.395

    def __init__(self, port: str, baudrate: int = 115200,
                 screen_w: int = 1920, screen_h: int = 1080):
        self._port     = port
        self._baudrate = baudrate
        self._serial   = None
        self._connected = False
        self._lock     = threading.Lock()

        self.SCREEN_W = screen_w
        self.SCREEN_H = screen_h

        # 피코 커서 추적 (프레임 좌표 기준, 리셋 후 중앙에서 시작)
        self._cur_x = screen_w // 2
        self._cur_y = screen_h // 2

    # ── 연결 ──────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baudrate, timeout=1.0)
            time.sleep(0.5)
            self._connected = True
            print(f"[PicoController] 연결: {self._port} @ {self._baudrate}")
            self._reset_cursor()
            return True
        except Exception as e:
            print(f"[PicoController] 연결 실패: {e}")
            return False

    def disconnect(self):
        try:
            self._send("RELEASE")
            time.sleep(0.05)
        except Exception:
            pass
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False
        print("[PicoController] 연결 해제")

    # ── 커서 리셋 ─────────────────────────────────────────────────
    def _reset_cursor(self):
        """
        피코 커서를 전체 좌상단(0,0)으로 보낸 뒤
        게임 화면 중앙(SCREEN_W//2, SCREEN_H//2)으로 이동.

        게임 모니터가 가장 왼쪽(left 최솟값)에 있으면:
          피코(0,0) = 게임 좌상단
          → 게임 중앙까지 SCREEN_W//2, SCREEN_H//2 만큼 이동
          → 이후 프레임 좌표 = 피코 추적 좌표 (1:1 대응)
        """
        self._send("MOVE:-9999:-9999")
        time.sleep(0.8)

        cx = self.SCREEN_W // 2
        cy = self.SCREEN_H // 2
        sx = round(cx * self.SCALE_X)
        sy = round(cy * self.SCALE_Y)
        self._send(f"MOVE:{sx}:{sy}")
        time.sleep(0.5)

        # 리셋 후 추적 기준점 = 게임 화면 중앙
        self._cur_x = cx
        self._cur_y = cy
        print(f"[PicoController] 커서 리셋 완료: 게임중앙({cx},{cy}) 전송({sx},{sy})")

    # ── 드래그 공격 ───────────────────────────────────────────────
    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 8, drag_dy: int = 0,
                    hold_ms: int = 80):
        """
        프레임 좌표(x, y)로 이동 → PRESS → 드래그 → RELEASE.

        x, y : YOLO 탐지 결과 좌표 (프레임 내, 0~SCREEN_W)
               피코 리셋 후 _cur_x/y 와 동일한 좌표계
        """
        # 1. 현재 추적위치 → 목표로 이동
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = round(dx * self.SCALE_X)
        sdy = round(dy * self.SCALE_Y)

        print(f"[Pico] ({self._cur_x},{self._cur_y})→({x},{y}) "
              f"이동({dx},{dy}) 전송MOVE({sdx},{sdy})")

        if sdx != 0 or sdy != 0:
            self._send(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)

        self._cur_x = x
        self._cur_y = y

        # 2. PRESS
        self._send("PRESS")
        time.sleep(hold_ms / 1000.0)

        # 3. 드래그
        if drag_dx != 0 or drag_dy != 0:
            dsdx = round(drag_dx * self.SCALE_X)
            dsdy = round(drag_dy * self.SCALE_Y)
            self._send(f"MOVE:{dsdx}:{dsdy}")
            time.sleep(0.03)
            self._cur_x += drag_dx
            self._cur_y += drag_dy

        # 4. RELEASE
        self._send("RELEASE")
        print(f"[Pico] 공격완료 @ ({self._cur_x},{self._cur_y})")

    # ── 공격 클릭 (몬스터 공격용) ────────────────────────────────
    def attack_click(self, x: int, y: int, hold_ms: int = 80):
        """
        몬스터 공격 전용 클릭.
        MOVE로 목표 위치 이동 → CLICK (PRESS+대기+RELEASE) 전송.

        리니지 계열: 몬스터 위에 정확히 클릭 = 공격
        드래그(PRESS→MOVE→RELEASE)는 빈 공간 이동 명령으로 오인됨 → 사용 금지
        """
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = round(dx * self.SCALE_X)
        sdy = round(dy * self.SCALE_Y)

        print(f"[Pico] ({self._cur_x},{self._cur_y})→({x},{y}) "
              f"이동({dx},{dy}) 전송MOVE({sdx},{sdy})")

        if sdx != 0 or sdy != 0:
            self._send(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)

        self._cur_x = x
        self._cur_y = y

        self._send(f"CLICK:{hold_ms}")
        print(f"[Pico] 공격클릭 @ ({x},{y}) hold={hold_ms}ms")

    # ── 단순 클릭 (아데나 줍기용) ─────────────────────────────────
    def click(self, x: int, y: int, hold_ms: int = 50):
        dx = x - self._cur_x
        dy = y - self._cur_y
        sdx = round(dx * self.SCALE_X)
        sdy = round(dy * self.SCALE_Y)

        if sdx != 0 or sdy != 0:
            self._send(f"MOVE:{sdx}:{sdy}")
            time.sleep(0.05)

        self._cur_x = x
        self._cur_y = y
        self._send(f"CLICK:{hold_ms}")
        print(f"[Pico] 클릭: ({x},{y}) 전송MOVE({sdx},{sdy})")

    # ── 유틸 ──────────────────────────────────────────────────────
    def stop(self):
        self._send("RELEASE")
        self._send("STOP")

    def ping(self) -> bool:
        self._send("PING")
        try:
            return self._serial.readline().decode("utf-8", "ignore").strip() == "PONG"
        except Exception:
            return False

    def _send(self, text: str):
        if not self._connected or self._serial is None:
            return
        try:
            with self._lock:
                self._serial.write((text + "\n").encode("utf-8"))
                self._serial.flush()
        except Exception as e:
            print(f"[PicoController] 전송 실패: {e}")
            self._connected = False

    # 하위 호환
    def click_drag(self, x, y, drag_dx=8, drag_dy=0, hold_ms=80):
        self.drag_attack(x, y, drag_dx, drag_dy, hold_ms)

    @property
    def is_connected(self):
        return self._connected
