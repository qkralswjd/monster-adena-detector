"""
controller.py
-------------
피코 HID 컨트롤러.

프로토콜:
  PING          → PONG
  MOVE:dx:dy    → 상대 마우스 이동
  PRESS         → 왼쪽 버튼 누름
  RELEASE       → 왼쪽 버튼 놓음
  CLICK[:ms]    → PRESS + 대기 + RELEASE

SCALE = 0.395  (MOVE:1 → 실제 2.53px 이동)

커서 기준:
  매 공격마다:
    1. MOVE:-9999:-9999  → 커서를 (0,0) 으로
    2. MOVE:+cx*S:+cy*S  → 게임 중앙으로 이동
    3. 중앙 기준 몬스터까지 상대이동
    4. PRESS + RELEASE
  → 오차 누적 없음, 항상 중앙 기준으로 정확하게 계산
"""

import time
import threading
import ctypes


class PicoController:

    SCALE = 0.395  # HID 감도 스케일

    def __init__(self, port: str, baudrate: int = 115200,
                 mon_left: int = 0, mon_top: int = 0,
                 lb_x: int = 0,
                 game_w: int = 1440, game_h: int = 1080):
        self._port      = port
        self._baudrate  = baudrate
        self._ser       = None
        self._lock      = threading.Lock()
        self._connected = False

        self._mon_left = mon_left
        self._mon_top  = mon_top
        self._lb_x     = lb_x

        # 게임 중앙 (기준점) - 게임 영역 내 픽셀
        self._center_x = game_w // 2   # 720
        self._center_y = game_h // 2   # 540

        # 공격 스레드 상태
        self._attacking = False

    # ── 연결 / 해제 ───────────────────────────────
    def connect(self) -> bool:
        try:
            import serial
            self._ser = serial.Serial(self._port, self._baudrate, timeout=1.0)
            time.sleep(0.5)
            self._connected = True
            print(f"[Pico] 연결: {self._port}")
            self._reset()
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

    # ── 커서 리셋 (connect 시 1회) ────────────────
    def _reset(self):
        """연결 시 1회만 호출. 피코 커서를 (0,0)으로 초기화."""
        ctypes.windll.user32.SetCursorPos(0, 0)
        time.sleep(0.1)
        self._send("MOVE:-9999:-9999")
        time.sleep(1.5)
        print(f"[Pico] 초기 리셋 완료 → (0,0)")

    def _move_to_center(self):
        """
        매 공격 전 호출.
        (0,0) → 게임 중앙으로 이동.
        항상 (0,0) 기준에서 출발하므로 오차 누적 없음.
        """
        # 1. 좌상단으로
        self._send("MOVE:-9999:-9999")
        time.sleep(0.08)
        # 2. 게임 중앙까지 절대이동 (Windows 원점 기준)
        # 게임 중앙의 Windows 좌표 = mon_left + lb_x + center_x, mon_top + center_y
        abs_cx = self._mon_left + self._lb_x + self._center_x
        abs_cy = self._mon_top  + self._center_y
        dx = round(abs_cx * self.SCALE)
        dy = round(abs_cy * self.SCALE)
        self._send(f"MOVE:{dx}:{dy}")
        time.sleep(0.05)

    # ── 드래그 공격 (비동기) ──────────────────────
    def drag_attack(self, gx: int, gy: int,
                    drag_dx: int = 0, drag_dy: int = 30,
                    hold_ms: int = 80):
        """
        gx, gy = 게임 영역 내 픽셀 좌표 (YOLO cx - lb_x, cy)
        매 공격마다 중앙으로 리셋 후 중앙 기준 상대이동 → 오차 없음.
        별도 스레드로 실행.
        """
        if self._attacking:
            return
        t = threading.Thread(
            target=self._attack_thread,
            args=(gx, gy, drag_dx, drag_dy, hold_ms),
            daemon=True
        )
        t.start()

    def _attack_thread(self, gx, gy, drag_dx, drag_dy, hold_ms):
        self._attacking = True
        try:
            # 1. 매번 중앙으로 리셋 (오차 누적 방지)
            self._move_to_center()

            # 2. 중앙 → 몬스터까지 상대이동
            #    중앙 기준 오프셋
            rel_x = gx - self._center_x
            rel_y = gy - self._center_y
            sdx   = round(rel_x * self.SCALE)
            sdy   = round(rel_y * self.SCALE)
            print(f"[Pico] 중앙({self._center_x},{self._center_y}) → 몬스터({gx},{gy})  상대({rel_x},{rel_y})  HID({sdx},{sdy})")
            if sdx != 0 or sdy != 0:
                self._send(f"MOVE:{sdx}:{sdy}")
                time.sleep(0.05)

            # 3. PRESS
            self._send("PRESS")
            time.sleep(hold_ms / 1000.0)

            # 4. 드래그
            if drag_dx != 0 or drag_dy != 0:
                dsdx = round(drag_dx * self.SCALE)
                dsdy = round(drag_dy * self.SCALE)
                self._send(f"MOVE:{dsdx}:{dsdy}")
                time.sleep(0.03)

            # 5. RELEASE
            self._send("RELEASE")


        except Exception as e:
            print(f"[Pico] 오류: {e}")
            try:
                self._send("RELEASE")
            except Exception:
                pass
        finally:
            self._attacking = False

    # ── 전송 ──────────────────────────────────────
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


class DummyController:
    """피코 미연결 시 로그만 출력"""

    def __init__(self):
        self._connected = False
        self.is_attacking = False

    def connect(self) -> bool:
        self._connected = True
        print("[Dummy] 연결 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False

    def drag_attack(self, sc_x, sc_y, drag_dx=0, drag_dy=30, hold_ms=80):
        print(f"[Dummy] 드래그공격: 전체화면({sc_x},{sc_y})")

    def stop(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_attacking(self) -> bool:
        return False
