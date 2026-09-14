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
  connect() 시 MOVE:-9999:-9999 → 커서를 Windows(0,0)으로 이동
  이후 SetCursorPos(game_origin_x, game_origin_y) 로 게임 영역 좌상단으로 이동
  _cur_x/_cur_y = 게임 영역 내 픽셀 좌표 (0,0 = 게임 좌상단)
  drag_attack(gx, gy) 인자도 게임 영역 내 픽셀 좌표
"""

import time
import threading
import ctypes


class PicoController:

    SCALE = 0.395  # HID 감도 스케일

    def __init__(self, port: str, baudrate: int = 115200,
                 mon_left: int = 0, mon_top: int = 0,
                 lb_x: int = 0):
        self._port     = port
        self._baudrate = baudrate
        self._ser      = None
        self._lock     = threading.Lock()
        self._connected = False

        self._mon_left = mon_left
        self._mon_top  = mon_top
        self._lb_x     = lb_x

        # 게임 영역 좌상단의 Windows 절대 좌표
        self._game_origin_x = mon_left + lb_x
        self._game_origin_y = mon_top

        # 피코 커서 현재 위치 (게임 영역 내 픽셀, 리셋 후 0,0 = 게임 좌상단)
        self._cur_x = 0
        self._cur_y = 0

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
        # 1. Windows 커서를 게임 영역 좌상단으로 이동
        ox = self._game_origin_x
        oy = self._game_origin_y
        ctypes.windll.user32.SetCursorPos(ox, oy)
        time.sleep(0.1)
        # 2. 피코도 같은 위치로 맞춤 (MOVE:-9999:-9999 후 게임좌상단으로 이동)
        self._send("MOVE:-9999:-9999")
        time.sleep(1.5)
        # 3. (0,0) → 게임 좌상단까지 이동
        dx = round(ox * self.SCALE)
        dy = round(oy * self.SCALE)
        if dx != 0 or dy != 0:
            self._send(f"MOVE:{dx}:{dy}")
            time.sleep(0.3)
        self._cur_x = 0
        self._cur_y = 0
        print(f"[Pico] 커서 리셋 완료 → 게임좌상단({ox},{oy}) = 게임내(0,0)")

    # ── 드래그 공격 (비동기) ──────────────────────
    def drag_attack(self, sc_x: int, sc_y: int,
                    drag_dx: int = 0, drag_dy: int = 30,
                    hold_ms: int = 80):
        """
        sc_x, sc_y = 게임 영역 내 픽셀 좌표 (0,0 = 게임 좌상단)
                     = YOLO 탐지 cx - lb_x,  cy
        별도 스레드로 실행 → 메인루프 블로킹 없음
        """
        if self._attacking:
            return
        t = threading.Thread(
            target=self._attack_thread,
            args=(sc_x, sc_y, drag_dx, drag_dy, hold_ms),
            daemon=True
        )
        t.start()

    def _attack_thread(self, sc_x, sc_y, drag_dx, drag_dy, hold_ms):
        self._attacking = True
        try:
            # 현재 위치 → 목표 위치 상대이동
            dx  = sc_x - self._cur_x
            dy  = sc_y - self._cur_y
            sdx = round(dx * self.SCALE)
            sdy = round(dy * self.SCALE)
            if sdx != 0 or sdy != 0:
                self._send(f"MOVE:{sdx}:{sdy}")
                time.sleep(0.05)
            self._cur_x = sc_x
            self._cur_y = sc_y

            # PRESS
            self._send("PRESS")
            time.sleep(hold_ms / 1000.0)

            # 드래그
            if drag_dx != 0 or drag_dy != 0:
                dsdx = round(drag_dx * self.SCALE)
                dsdy = round(drag_dy * self.SCALE)
                self._send(f"MOVE:{dsdx}:{dsdy}")
                self._cur_x += drag_dx
                self._cur_y += drag_dy
                time.sleep(0.03)

            # RELEASE
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
