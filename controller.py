"""
controller.py
-------------
PICO HID 절대좌표 클릭 컨트롤러 — pico_image_autoclicker 방식 적용.

변경 내용 (오픈루프 → 클로즈드루프):
  - _move_to(): 최대 MAX_CORRECTION_ITERS 회 반복, ACK 수신 후 재측정
  - CORRECTION_DAMPING=0.35: Windows 포인터 가속(1.4~2.5x) 발산 방지
  - CORRECTION_TOLERANCE_PX=4: 4px 이내 수렴 시 완료
  - CLICK:ms 원자 명령: PRESS + sleep + RELEASE → Pico 내부 처리
  - ACK 대기: _wait_ack() — OK:MOVE / OK:CLICK / OK:PRESS / OK:RELEASE
  - PING/PONG 하트비트: ping() 메서드

PICO 프로토콜:
  MOVE:dx:dy     → OK:MOVE / ERR:MOVE
  CLICK[:ms]     → OK:CLICK / ERR:CLICK  (press+wait+release 원자 처리)
  PRESS          → OK:PRESS
  RELEASE        → OK:RELEASE
  PING           → PONG
  STOP           → OK:STOP
"""

import time
import threading
import ctypes
import ctypes.wintypes

# ── 클로즈드루프 파라미터 (pico_image_autoclicker 동일 값) ────────────
CORRECTION_TOLERANCE_PX = 4    # 이 이내면 수렴 완료
MAX_CORRECTION_ITERS    = 20   # 최대 반복 횟수
CORRECTION_DAMPING      = 0.35 # 35% 감쇠: Windows 포인터 가속 발산 방지
ACK_TIMEOUT_S           = 0.3  # ACK 대기 타임아웃


def _get_cursor_pos():
    """Windows 실제 커서 위치 반환 (read-only, 입력 주입 없음)."""
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


class PicoController:

    def __init__(self, port: str, baudrate: int = 115200,
                 click_pulse_ms: int = 20):
        self._port          = port
        self._baudrate      = baudrate
        self._click_pulse_ms = click_pulse_ms
        self._ser           = None
        self._lock          = threading.Lock()
        self._connected     = False
        self._attacking     = False
        self._rx_buffer     = b""

    # ── 연결 ──────────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            import serial
            self._ser = serial.Serial(self._port, self._baudrate, timeout=0)
            time.sleep(0.5)
            self._connected = True
            print(f"[Pico] 연결: {self._port} @ {self._baudrate}")
            return True
        except Exception as e:
            print(f"[Pico] 연결 실패: {e}")
            return False

    def disconnect(self):
        try:
            self._write_line("RELEASE")
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

    # ── 저수준 전송 ───────────────────────────────────────────────────
    def _write_line(self, text: str):
        if not self._connected or self._ser is None:
            return
        try:
            with self._lock:
                self._ser.write((text + "\n").encode("utf-8"))
                self._ser.flush()
        except Exception as e:
            print(f"[Pico] 전송 실패: {e}")
            self._connected = False

    # ── ACK 대기 (pico_image_autoclicker._wait_for_ack 동일) ──────────
    def _wait_ack(self, ok_line: str, err_line: str = None) -> bool:
        """
        ok_line 수신 시 True, err_line 수신 시 False, 타임아웃 시 None 반환.
        PONG은 투명하게 처리.
        """
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                waiting = self._ser.in_waiting if self._ser else 0
            except Exception:
                return False
            if waiting:
                self._rx_buffer += self._ser.read(waiting)
                while b"\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    if text == "PONG":
                        continue  # 하트비트 투명 처리
                    if text == ok_line:
                        return True
                    if err_line and text == err_line:
                        print(f"[Pico] ERR 응답: {text}")
                        return False
            time.sleep(0.005)
        print(f"[Pico] ACK 타임아웃 ({ok_line})")
        return None  # 타임아웃

    # ── 클로즈드루프 이동 (pico_image_autoclicker._move_to 동일) ──────
    def _move_to(self, target_x: int, target_y: int) -> bool:
        """
        GetCursorPos()로 현재 위치 측정 → MOVE 전송 → ACK → 재측정 → 반복.
        오차 4px 이내 수렴 또는 최대 20회 시 종료.
        """
        for i in range(MAX_CORRECTION_ITERS):
            cur_x, cur_y = _get_cursor_pos()
            dx = target_x - cur_x
            dy = target_y - cur_y

            if abs(dx) <= CORRECTION_TOLERANCE_PX and abs(dy) <= CORRECTION_TOLERANCE_PX:
                print(f"[Pico] 수렴 완료 ({i}회)  실제=({cur_x},{cur_y})  오차=({dx},{dy})")
                return True

            # 35% 감쇠: 과보정 방지 (Windows 포인터 가속 대응)
            sdx = round(dx * CORRECTION_DAMPING) or (1 if dx > 0 else -1)
            sdy = round(dy * CORRECTION_DAMPING) or (1 if dy > 0 else -1)

            print(f"[Pico] 이동 {i+1}회: ({cur_x},{cur_y})→({target_x},{target_y})  "
                  f"Δ({dx},{dy})  HID({sdx},{sdy})")

            self._write_line(f"MOVE:{sdx}:{sdy}")
            self._wait_ack("OK:MOVE", "ERR:MOVE")
            time.sleep(0.02)  # OS 커서 위치 안정화 대기

        # 수렴 실패 (드물게 발생)
        cur_x, cur_y = _get_cursor_pos()
        print(f"[Pico] 수렴 실패  최종=({cur_x},{cur_y})  "
              f"오차=({cur_x - target_x},{cur_y - target_y})")
        return False

    # ── CLICK 원자 명령 (Pico 내부에서 press+wait+release) ────────────
    def _click(self, pulse_ms: int = None) -> bool:
        """
        CLICK[:ms] 전송 → Pico가 press + sleep(ms) + release 처리 → OK:CLICK.
        PC에서 sleep 하지 않음.
        """
        ms = pulse_ms if pulse_ms is not None else self._click_pulse_ms
        self._write_line(f"CLICK:{ms}")
        ok = self._wait_ack("OK:CLICK", "ERR:CLICK")
        if ok:
            print(f"[Pico] CLICK 완료 ({ms}ms)")
        return bool(ok)

    # ── 공격: 이동 + CLICK (비동기) ───────────────────────────────────
    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 0, drag_dy: int = 0,
                    hold_ms: int = 80):
        """
        main.py 호환용 메서드 이름 유지.
        실제 동작: 클로즈드루프 이동 → CLICK 원자 명령.
        drag_dx/dy 는 무시 (단순 클릭으로 통일).
        """
        if self._attacking:
            return
        t = threading.Thread(
            target=self._attack_thread,
            args=(x, y, hold_ms),
            daemon=True
        )
        t.start()

    def _attack_thread(self, x: int, y: int, hold_ms: int):
        self._attacking = True
        try:
            # 1. 클로즈드루프 이동
            self._move_to(x, y)

            # 2. CLICK 원자 명령 (Pico 내부 처리)
            self._click(hold_ms)

            print(f"[Pico] 공격 완료: ({x},{y})")

        except Exception as e:
            print(f"[Pico] 오류: {e}")
            try:
                self._write_line("RELEASE")  # 안전장치
            except Exception:
                pass
        finally:
            self._attacking = False

    # ── 단순 클릭 ─────────────────────────────────────────────────────
    def click(self, x: int, y: int, pulse_ms: int = None):
        """클로즈드루프 이동 → CLICK 원자 명령."""
        self._move_to(x, y)
        self._click(pulse_ms)

    # ── 드래그 (필요 시 사용) ──────────────────────────────────────────
    def drag(self, from_x: int, from_y: int, to_x: int, to_y: int,
             steps: int = 8, hold_ms: int = 50):
        """
        pico_image_autoclicker._do_drag 동일 방식:
        클로즈드루프로 시작점 이동 → PRESS → 웨이포인트 이동 → RELEASE.
        """
        if not self._move_to(from_x, from_y):
            print(f"[Pico] 드래그 시작점 미도달: ({from_x},{from_y})")
            return

        self._write_line("PRESS")
        if not self._wait_ack("OK:PRESS", "ERR:PRESS"):
            print("[Pico] PRESS 실패")
            return
        time.sleep(hold_ms / 1000.0)

        steps = max(1, steps)
        for i in range(1, steps + 1):
            wx = round(from_x + (to_x - from_x) * i / steps)
            wy = round(from_y + (to_y - from_y) * i / steps)
            is_last = (i == steps)
            self._move_to(wx, wy)
            if not is_last:
                time.sleep(0.03)

        self._write_line("RELEASE")
        self._wait_ack("OK:RELEASE", "ERR:RELEASE")
        print(f"[Pico] 드래그 완료: ({from_x},{from_y})→({to_x},{to_y})")

    # ── 하트비트 ──────────────────────────────────────────────────────
    def ping(self) -> bool:
        """PING 전송 → PONG 확인. 연결 상태 점검용."""
        self._write_line("PING")
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                waiting = self._ser.in_waiting if self._ser else 0
            except Exception:
                return False
            if waiting:
                self._rx_buffer += self._ser.read(waiting)
                while b"\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text == "PONG":
                        return True
            time.sleep(0.005)
        return False

    def stop(self):
        self._write_line("STOP")
        self._attacking = False

    # ── 긴급 중단 ─────────────────────────────────────────────────────
    def emergency_stop(self):
        """버튼 강제 해제."""
        self._write_line("RELEASE")
        self._attacking = False


# ── Dummy (PICO 미연결 시) ─────────────────────────────────────────────
class DummyController:
    """실제 입력 없이 로그만 출력. 좌표 확인용."""

    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        print("[Dummy] 연결 (실제 입력 없음)")
        return True

    def disconnect(self):
        self._connected = False

    def drag_attack(self, x: int, y: int,
                    drag_dx: int = 0, drag_dy: int = 0,
                    hold_ms: int = 80):
        print(f"[Dummy] drag_attack: ({x},{y})  hold={hold_ms}ms")

    def click(self, x: int, y: int, pulse_ms: int = None):
        print(f"[Dummy] click: ({x},{y})  pulse={pulse_ms}ms")

    def drag(self, from_x: int, from_y: int, to_x: int, to_y: int,
             steps: int = 8, hold_ms: int = 50):
        print(f"[Dummy] drag: ({from_x},{from_y})→({to_x},{to_y})")

    def ping(self) -> bool:
        return True

    def stop(self):
        pass

    def emergency_stop(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_attacking(self) -> bool:
        return False
