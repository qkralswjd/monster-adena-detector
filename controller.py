"""
controller.py
-------------
PICO HID 절대좌표 클릭 컨트롤러.

좌표 원칙:
  - 입력: 화면 절대좌표 (x, y)  ← detector.py cx/cy 그대로
  - 좌표 변환 없음
  - lb_x / lb_y 보정 없음
  - 중앙 이동 없음
  - 상대 이동 없음

동작 원리:
  1. 연결 시 MOVE:-9999:-9999 → PICO 커서를 화면 (0,0) 으로 리셋
  2. 리셋 후 _cur_x=0, _cur_y=0 으로 초기화
  3. click(x, y) 호출 시:
       dx = x - _cur_x   (목표 - 현재)
       dy = y - _cur_y
       MOVE: round(dx * SCALE) : round(dy * SCALE)
       _cur_x = x, _cur_y = y  (추적 갱신)
  4. SCALE = 0.395  (MOVE:1 → 실제 2.53px)

PICO 프로토콜:
  MOVE:dx:dy   → 상대 마우스 이동
  PRESS        → 왼쪽 버튼 누름
  RELEASE      → 왼쪽 버튼 놓음
  PING         → PONG 응답 (연결 확인용)
"""

import time
import threading
import ctypes
import ctypes.wintypes


def _get_cursor_pos():
    """Windows 실제 커서 위치 반환."""
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


class PicoController:

    SCALE_X = 0.3955  # 측정값 (3지점 평균)
    SCALE_Y = 0.3899  # 측정값 (3지점 평균)

    def __init__(self, port: str, baudrate: int = 115200):
        self._port      = port
        self._baudrate  = baudrate
        self._ser       = None
        self._lock      = threading.Lock()
        self._connected = False
        self._attacking = False

    # ── 연결 ──────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            import serial
            self._ser = serial.Serial(self._port, self._baudrate, timeout=1.0)
            time.sleep(0.5)
            self._connected = True
            print(f"[Pico] 연결: {self._port} @ {self._baudrate}")
            return True
        except Exception as e:
            print(f"[Pico] 연결 실패: {e}")
            return False

    def disconnect(self):
        try:
            self._send("RELEASE")
        except Exception:
            pass
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False
        self._attacking = False
        print("[Pico] 연결 해제")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_attacking(self) -> bool:
        return self._attacking

    # ── 절대좌표 이동 ──────────────────────────────────────────
    def _move_to(self, x: int, y: int):
        """
        GetCursorPos() 로 실제 커서 위치 읽어서 상대이동.
        리셋/추적 불필요. 항상 정확.
        """
        cur_x, cur_y = _get_cursor_pos()
        dx = x - cur_x
        dy = y - cur_y
        sdx = round(dx * self.SCALE_X)
        sdy = round(dy * self.SCALE_Y)

        print(f"[Pico] 이동: ({cur_x},{cur_y}) → ({x},{y})"
              f"  Δ({dx},{dy})  HID({sdx},{sdy})")

        if sdx != 0 or sdy != 0:
            self._send(f"MOVE:{sdx}:{sdy}")

    # ── 드래그 공격 (비동기) ──────────────────────────────────
    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 0, drag_dy: int = 30,
                    hold_ms: int = 80):
        """
        (x, y) 로 이동 → PRESS → 드래그 → RELEASE.
        x, y = 화면 절대좌표. 변환 없음.
        비동기 스레드로 실행.
        """
        if self._attacking:
            return
        t = threading.Thread(
            target=self._attack_thread,
            args=(x, y, drag_dx, drag_dy, hold_ms),
            daemon=True
        )
        t.start()

    def _attack_thread(self, x, y, drag_dx, drag_dy, hold_ms):
        self._attacking = True
        try:
            # 1. 목표 좌표로 이동
            self._move_to(x, y)

            # 2. PRESS
            self._send("PRESS")
            time.sleep(hold_ms / 1000.0)

            # 3. 드래그
            if drag_dx != 0 or drag_dy != 0:
                sdx = round(drag_dx * self.SCALE_X)
                sdy = round(drag_dy * self.SCALE_Y)
                self._send(f"MOVE:{sdx}:{sdy}")
                time.sleep(0.03)

            # 4. RELEASE
            self._send("RELEASE")
            print(f"[Pico] 공격 완료: ({x},{y})")

        except Exception as e:
            print(f"[Pico] 오류: {e}")
            try:
                self._send("RELEASE")
            except Exception:
                pass
        finally:
            self._attacking = False

    # ── 단순 클릭 ─────────────────────────────────────────────
    def click(self, x: int, y: int, hold_ms: int = 50):
        """
        (x, y) 로 이동 → PRESS → RELEASE.
        x, y = 화면 절대좌표. 변환 없음.
        """
        self._move_to(x, y)
        self._send("PRESS")
        time.sleep(hold_ms / 1000.0)
        self._send("RELEASE")
        print(f"[Pico] 클릭: ({x},{y})")

    # ── 전송 ──────────────────────────────────────────────────
    def _send(self, text: str):
        if not self._connected or self._ser is None:
            return
        try:
            with self._lock:
                self._ser.write((text + "\n").encode())
                self._ser.flush()
        except Exception as e:
            print(f"[Pico] 전송 실패: {e}")
            self._connected = False

    def stop(self):
        self._send("RELEASE")
        self._attacking = False

    def ping(self) -> bool:
        self._send("PING")
        try:
            resp = self._ser.readline().decode("utf-8", "ignore").strip()
            return resp == "PONG"
        except Exception:
            return False


# ── Dummy (PICO 미연결 시) ────────────────────────────────────
class DummyController:
    """실제 입력 없이 로그만 출력. 좌표 확인용."""

    def __init__(self):
        self._connected = False
        self._cur_x = 0
        self._cur_y = 0

    def connect(self) -> bool:
        self._connected = True
        print("[Dummy] 연결 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False

    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 0, drag_dy: int = 30,
                    hold_ms: int = 80):
        print(f"[Dummy] drag_attack: ({x},{y})"
              f"  drag=({drag_dx},{drag_dy})  hold={hold_ms}ms")
        self._cur_x = x + drag_dx
        self._cur_y = y + drag_dy

    def click(self, x: int, y: int, hold_ms: int = 50):
        print(f"[Dummy] click: ({x},{y})  hold={hold_ms}ms")
        self._cur_x = x
        self._cur_y = y

    def stop(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_attacking(self) -> bool:
        return False
