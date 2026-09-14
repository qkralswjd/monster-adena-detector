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
import json
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
    Raspberry Pi Pico를 시리얼로 연결해서 HID 입력을 전송한다.

    Pico 쪽 펌웨어는 JSON 명령을 받아서 처리하도록 구성한다:
        {"cmd": "click", "x": 500, "y": 300, "btn": "left"}
        {"cmd": "move", "x": 500, "y": 300}

    연결 실패 시 DummyController로 폴백한다.
    """

    def __init__(self, port: str, baudrate: int = 115200):
        self._port = port
        self._baudrate = baudrate
        self._serial = None
        self._connected = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(
                self._port,
                self._baudrate,
                timeout=1.0
            )
            time.sleep(0.5)  # Pico 리셋 대기
            self._connected = True
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

    def attack(self, x: int, y: int):
        """
        지정 좌표에 마우스 클릭을 전송한다.
        """
        cmd = {"cmd": "click", "x": x, "y": y, "btn": "left"}
        self._send(cmd)

    def click_drag(self, x: int, y: int, drag_dx: int = 5, drag_dy: int = 0, hold_ms: int = 100):
        """
        몬스터 좌표에 클릭 드래그 → 자동공격 발동.
        1) mousedown (x, y)
        2) hold_ms 대기
        3) 살짝 drag (x+drag_dx, y+drag_dy)
        4) mouseup
        """
        self._send({"cmd": "mousedown", "x": x, "y": y, "btn": "left"})
        time.sleep(hold_ms / 1000.0)
        self._send({"cmd": "mouseup", "x": x + drag_dx, "y": y + drag_dy, "btn": "left"})

    def click_move(self, x: int, y: int):
        """
        바닥 좌표 클릭 → 캐릭터 이동.
        단순 클릭 1회.
        """
        cmd = {"cmd": "click", "x": x, "y": y, "btn": "left"}
        self._send(cmd)

    def move(self, x: int, y: int):
        """마우스 커서 이동만 전송한다 (클릭 없음)."""
        cmd = {"cmd": "move", "x": x, "y": y}
        self._send(cmd)

    def key_press(self, key: str):
        """키 입력을 전송한다. key는 예: 'a', 'space', 'f1'."""
        cmd = {"cmd": "key", "key": key}
        self._send(cmd)

    def _send(self, cmd: dict):
        """JSON 직렬화 후 시리얼로 전송한다."""
        if not self._connected or self._serial is None:
            return
        try:
            data = (json.dumps(cmd) + "\n").encode("utf-8")
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
