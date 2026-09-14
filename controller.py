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

공격 방식:
  매 프레임 탐지된 최신 좌표로 MOVE 추적 → PRESS → 아래 드래그 → RELEASE
  drag_attack은 별도 스레드로 실행 → 메인 루프 블로킹 없음
  공격 중에도 탐지/좌표 갱신 계속
"""

import time
import threading
import ctypes
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
        self.is_attacking = False

    def connect(self):
        self._connected = True
        print("[DummyController] 연결 완료 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False

    def drag_attack(self, x, y, drag_dx=0, drag_dy=30, hold_ms=80):
        print(f"[DummyController] 드래그공격: ({x},{y}) drag=({drag_dx},{drag_dy})")

    def attack_click(self, x, y, hold_ms=80):
        print(f"[DummyController] 공격클릭: ({x},{y}) hold={hold_ms}ms")

    def update_target(self, x, y):
        """공격 중 최신 좌표 갱신 (Dummy는 무시)"""
        pass

    def click(self, x, y, hold_ms=50):
        print(f"[DummyController] 클릭: ({x},{y})")

    def stop(self):
        pass

    def click_drag(self, x, y, drag_dx=0, drag_dy=30, hold_ms=80):
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
                 screen_w: int = 1920, screen_h: int = 1080,
                 mon_left: int = 0, mon_top: int = 0):
        self._port     = port
        self._baudrate = baudrate
        self._serial   = None
        self._connected = False
        self._lock     = threading.Lock()

        self.SCREEN_W = screen_w
        self.SCREEN_H = screen_h
        self.MON_LEFT = mon_left   # 게임모니터 left (예: -1920)
        self.MON_TOP  = mon_top    # 게임모니터 top  (보통 0)

        # 피코 커서 추적 ─ 전체화면 절대좌표 기준
        # 리셋 후 게임모니터 좌상단(mon_left, mon_top) = 피코(0,0)
        # 전체화면 중앙 = mon_left + screen_w//2
        self._cur_x = mon_left + screen_w // 2
        self._cur_y = mon_top  + screen_h // 2

        # 공격 스레드 상태
        self._attack_thread: threading.Thread = None
        self._attacking = False          # 현재 공격 중 여부
        self._attack_lock = threading.Lock()

        # 공격 중 최신 타겟 좌표 (메인루프가 갱신)
        self._target_x = mon_left + screen_w // 2
        self._target_y = mon_top  + screen_h // 2

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
        self._attacking = False
        print("[PicoController] 연결 해제")

    # ── 공격 중 여부 ──────────────────────────────────────────────
    @property
    def is_attacking(self) -> bool:
        return self._attacking

    # ── 공격 중 타겟 좌표 실시간 갱신 ────────────────────────────
    def update_target(self, x: int, y: int):
        """
        메인 루프에서 매 프레임 호출.
        공격 스레드가 PRESS 상태일 때 최신 몬스터 좌표를 반영해
        MOVE로 커서를 따라가게 함.
        """
        with self._attack_lock:
            self._target_x = x
            self._target_y = y

        # 공격 중이면 즉시 커서를 최신 위치로 추적
        if self._attacking:
            dx = x - self._cur_x
            dy = y - self._cur_y
            sdx = round(dx * self.SCALE_X)
            sdy = round(dy * self.SCALE_Y)
            if sdx != 0 or sdy != 0:
                self._send(f"MOVE:{sdx}:{sdy}")
                self._cur_x = x
                self._cur_y = y

    # ── 커서 리셋 ─────────────────────────────────────────────────
    def _reset_cursor(self):
        """
        1. ctypes로 Windows 실제 커서를 (0,0)으로 강제 이동
        2. 피코도 MOVE:-9999:-9999 로 (0,0) 리셋
        → 피코 커서 = Windows 커서 = (0,0) 완전 동기화
        → 이후 전체화면 절대좌표로 dx 계산하면 정확히 맞음
        """
        # Windows 커서를 (0,0)으로 강제 이동
        ctypes.windll.user32.SetCursorPos(0, 0)
        time.sleep(0.1)

        # 피코도 (0,0)으로 리셋
        self._send("MOVE:-9999:-9999")
        time.sleep(0.8)

        self._cur_x = 0
        self._cur_y = 0
        print(f"[PicoController] 커서 리셋 완료: Windows+피코 모두 (0,0) 동기화")

    # ── 드래그 공격 (비동기 스레드) ──────────────────────────────
    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 0, drag_dy: int = 30,
                    hold_ms: int = 80):
        """
        몬스터 좌표로 이동 → PRESS → 드래그 → RELEASE
        공격 중이면 무시 (새 공격 시작 안 함)
        """
        if self._attacking:
            return

        t = threading.Thread(
            target=self._drag_attack_thread,
            args=(x, y, drag_dx, drag_dy, hold_ms),
            daemon=True
        )
        t.start()
        self._attack_thread = t

    def _drag_attack_thread(self, x: int, y: int,
                             drag_dx: int, drag_dy: int, hold_ms: int):
        """단순 클릭드래그: MOVE → PRESS → 드래그 → RELEASE"""
        self._attacking = True
        try:
            # 1. 몬스터 위치로 이동
            dx = x - self._cur_x
            dy = y - self._cur_y
            sdx = round(dx * self.SCALE_X)
            sdy = round(dy * self.SCALE_Y)
            if sdx != 0 or sdy != 0:
                self._send(f"MOVE:{sdx}:{sdy}")
                time.sleep(0.05)
            self._cur_x = x
            self._cur_y = y

            # 2. 클릭
            self._send("PRESS")
            time.sleep(hold_ms / 1000.0)

            # 3. 드래그 (아래로)
            if drag_dx != 0 or drag_dy != 0:
                dsdx = round(drag_dx * self.SCALE_X)
                dsdy = round(drag_dy * self.SCALE_Y)
                self._send(f"MOVE:{dsdx}:{dsdy}")
                time.sleep(0.03)
                self._cur_x += drag_dx
                self._cur_y += drag_dy

            # 4. 릴리즈
            self._send("RELEASE")
            print(f"[Pico] 클릭완료: ({x},{y})")

        except Exception as e:
            print(f"[PicoController] 오류: {e}")
            try:
                self._send("RELEASE")
            except Exception:
                pass
        finally:
            self._attacking = False

    # ── 공격 클릭 (단순 클릭 공격용) ─────────────────────────────
    def attack_click(self, x: int, y: int, hold_ms: int = 80):
        """MOVE → CLICK (PRESS+대기+RELEASE)"""
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
        self._attacking = False

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
    def click_drag(self, x, y, drag_dx=0, drag_dy=30, hold_ms=80):
        self.drag_attack(x, y, drag_dx, drag_dy, hold_ms)

    @property
    def is_connected(self):
        return self._connected
