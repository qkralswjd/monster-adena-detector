"""
controller.py
-------------
실제 입력(마우스/키보드) 전송만을 담당한다.

공격 로직과 입력 장치를 분리해서
나중에 Pico HID 외에 다른 장치로 바꿔도 이 파일만 수정하면 된다.

현재 지원:
  - PicoController  : Raspberry Pi Pico HID (시리얼 통신)
  - DummyController : 실제 입력 없이 로그만 출력 (테스트/개발용)

사용 예:
    controller = make_controller(cfg)
    controller.connect()
    controller.attack(x=500, y=300)
    controller.disconnect()
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional


# ------------------------------------------------------------------
# 추상 베이스 클래스
# ------------------------------------------------------------------

class BaseController(ABC):
    """모든 컨트롤러가 구현해야 하는 인터페이스."""

    @abstractmethod
    def connect(self) -> bool:
        """장치 연결. 성공이면 True."""
        ...

    @abstractmethod
    def disconnect(self):
        """장치 연결 해제."""
        ...

    @abstractmethod
    def attack(self, x: int, y: int):
        """
        지정한 화면 좌표에 공격 입력을 전송한다.
        x, y는 화면(스크린) 좌표다.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """현재 연결 상태."""
        ...


# ------------------------------------------------------------------
# Dummy Controller (테스트/개발용)
# ------------------------------------------------------------------

class DummyController(BaseController):
    """
    실제 입력 없이 로그만 출력하는 컨트롤러.
    Pico가 연결되지 않았을 때 자동으로 사용된다.
    """

    def __init__(self):
        self._connected = False
        self._attack_count = 0

    def connect(self) -> bool:
        self._connected = True
        print("[DummyController] 연결 완료 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False
        print("[DummyController] 연결 해제")

    def attack(self, x: int, y: int):
        self._attack_count += 1
        print(f"[DummyController] 공격 #{self._attack_count}: ({x}, {y})")

    def click_drag(self, x: int, y: int, drag_dx: int = 5, drag_dy: int = 0, hold_ms: int = 100):
        """클릭 드래그 (자동공격용) - Dummy는 로그만 출력."""
        print(f"[DummyController] 클릭드래그: ({x},{y}) → dx={drag_dx} hold={hold_ms}ms")

    def click_move(self, x: int, y: int):
        """바닥 클릭 이동 - Dummy는 로그만 출력."""
        print(f"[DummyController] 이동클릭: ({x}, {y})")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def attack_count(self) -> int:
        return self._attack_count


# ------------------------------------------------------------------
# Pico HID Controller
# ------------------------------------------------------------------

class PicoController(BaseController):
    """
    Raspberry Pi Pico HID 컨트롤러.

    펌웨어 프로토콜 (텍스트, 줄바꿈 종료):
        PING            → PONG
        MOVE:<dx>:<dy>  → 상대 이동 (현재 커서 기준)
        CLICK[:<ms>]    → 클릭 (기본 20ms)
        PRESS           → 마우스 누르기 (드래그용)
        RELEASE         → 마우스 떼기
        STOP            → 비상정지

    절대좌표 → 상대좌표 변환:
        PC에서 현재 커서 위치를 추적해서 차이값(dx,dy)으로 이동
    """

    def __init__(self, port: str, baudrate: int = 115200):
        self._port     = port
        self._baudrate = baudrate
        self._serial   = None
        self._connected = False
        self._lock     = threading.Lock()
        # 현재 커서 위치 추적 (절대→상대 변환용)
        self._cur_x = 0
        self._cur_y = 0

    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(
                self._port, self._baudrate, timeout=1.0)
            time.sleep(0.5)
            self._connected = True
            # 현재 커서 위치 초기화
            try:
                import ctypes
                pt = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                self._cur_x = pt.x
                self._cur_y = pt.y
            except Exception:
                self._cur_x = 0
                self._cur_y = 0
            print(f"[PicoController] 연결 완료: {self._port} @ {self._baudrate}")
            return True
        except Exception as e:
            print(f"[PicoController] 연결 실패: {e}")
            self._connected = False
            return False

    def disconnect(self):
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connected = False
        print("[PicoController] 연결 해제")

    def _get_cursor(self):
        """현재 실제 커서 위치 가져오기."""
        try:
            import ctypes
            pt = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y
        except Exception:
            return self._cur_x, self._cur_y

    def _move_to(self, abs_x: int, abs_y: int):
        """절대 좌표로 커서 이동 (상대 이동 명령으로 변환)."""
        cx, cy = self._get_cursor()
        dx = abs_x - cx
        dy = abs_y - cy
        if dx != 0 or dy != 0:
            self._send_text(f"MOVE:{dx}:{dy}")
            self._cur_x = abs_x
            self._cur_y = abs_y

    def attack(self, x: int, y: int):
        """절대 좌표로 이동 후 클릭."""
        self._move_to(x, y)
        time.sleep(0.02)
        self._send_text("CLICK:50")

    def click_drag(self, x: int, y: int, drag_dx: int = 5,
                   drag_dy: int = 0, hold_ms: int = 80):
        """
        절대 좌표로 이동 → PRESS → 살짝 드래그 → RELEASE.
        피코 펌웨어: PRESS / MOVE:<dx>:<dy> / RELEASE
        """
        self._move_to(x, y)
        time.sleep(0.02)
        self._send_text("PRESS")
        time.sleep(hold_ms / 1000.0)
        if drag_dx != 0 or drag_dy != 0:
            self._send_text(f"MOVE:{drag_dx}:{drag_dy}")
            self._cur_x += drag_dx
            self._cur_y += drag_dy
        self._send_text("RELEASE")

    def click_move(self, x: int, y: int):
        """절대 좌표로 이동 후 단순 클릭 (바닥 이동용)."""
        self._move_to(x, y)
        time.sleep(0.02)
        self._send_text("CLICK:20")

    def move(self, x: int, y: int):
        """절대 좌표로 커서만 이동 (클릭 없음)."""
        self._move_to(x, y)

    def key_press(self, key: str):
        """
        키 입력 — 피코 펌웨어가 키보드 HID를 지원하지 않으므로
        현재는 로그만 출력한다. 필요 시 펌웨어 확장 후 구현.
        """
        print(f"[PicoController] key_press({key!r}) — 펌웨어 미지원, 무시")

    def ping(self) -> bool:
        """PING → PONG 응답 확인. 연결 테스트용."""
        self._send_text("PING")
        try:
            resp = self._serial.readline().decode("utf-8", "ignore").strip()
            return resp == "PONG"
        except Exception:
            return False

    def stop(self):
        """비상 정지 (STOP 명령)."""
        self._send_text("STOP")

    def _send_text(self, text: str):
        """
        텍스트 명령을 줄바꿈(\n) 붙여 UTF-8로 시리얼 전송.
        피코 펌웨어가 기대하는 프로토콜 그대로 사용.
        """
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
    """
    config에 따라 적절한 컨트롤러를 생성해서 반환한다.
    controller.enabled가 False이거나 연결 실패 시 DummyController를 반환한다.
    """
    if not cfg.controller.enabled:
        print("[Controller] enabled=false → DummyController 사용")
        c = DummyController()
        c.connect()
        return c

    if cfg.controller.type == "pico":
        c = PicoController(
            port=cfg.controller.port,
            baudrate=cfg.controller.baudrate
        )
        if c.connect():
            return c
        else:
            print("[Controller] Pico 연결 실패 → DummyController로 폴백")
            dummy = DummyController()
            dummy.connect()
            return dummy

    # 알 수 없는 타입
    print(f"[Controller] 알 수 없는 타입: {cfg.controller.type} → DummyController")
    dummy = DummyController()
    dummy.connect()
    return dummy
