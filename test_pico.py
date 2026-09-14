"""
test_pico.py - 피코 단독 테스트
명령어:
  1. 리셋 테스트     : python test_pico.py reset
  2. 절대좌표 클릭   : python test_pico.py click 814 228
  3. 상대이동 테스트 : python test_pico.py move 100 50
"""
import sys
import time
import ctypes
import serial

PORT     = "COM4"
BAUDRATE = 115200
SCALE    = 0.395


def send(ser, text):
    print(f"  → 전송: {text}")
    ser.write((text + "\n").encode())
    ser.flush()


def connect():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1.0)
    time.sleep(0.5)
    print(f"[피코] 연결: {PORT}")
    return ser


def reset(ser):
    """커서를 (0,0)으로 리셋"""
    print("\n[리셋] Windows 커서 (0,0)으로 강제 이동")
    ctypes.windll.user32.SetCursorPos(0, 0)
    time.sleep(0.1)
    print("[리셋] MOVE:-9999:-9999 전송 (1.5초 대기)")
    send(ser, "MOVE:-9999:-9999")
    time.sleep(1.5)
    print("[리셋] 완료 → 피코(0,0) = Windows(0,0)")


def click_abs(ser, fx, fy, mon_left=-1920, mon_top=0):
    """
    프레임 좌표 → 전체화면 좌표 → 피코 클릭
    리셋(0,0) 기준 상대이동
    """
    sc_x = fx + mon_left
    sc_y = fy + mon_top
    sdx  = round(sc_x * SCALE)
    sdy  = round(sc_y * SCALE)

    print(f"\n[클릭] 프레임({fx},{fy})")
    print(f"       전체화면({sc_x},{sc_y})")
    print(f"       HID MOVE({sdx},{sdy})")

    if sdx != 0 or sdy != 0:
        send(ser, f"MOVE:{sdx}:{sdy}")
        time.sleep(0.05)

    send(ser, "PRESS")
    time.sleep(0.08)
    send(ser, "RELEASE")
    print("[클릭] 완료")


def move_rel(ser, dx, dy):
    """상대이동만 테스트"""
    sdx = round(dx * SCALE)
    sdy = round(dy * SCALE)
    print(f"\n[이동] 상대이동 ({dx},{dy}) → HID({sdx},{sdy})")
    send(ser, f"MOVE:{sdx}:{sdy}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("사용법:")
        print("  python test_pico.py reset")
        print("  python test_pico.py click <프레임x> <프레임y>")
        print("  python test_pico.py move <dx> <dy>")
        sys.exit()

    ser = connect()

    if args[0] == "reset":
        reset(ser)

    elif args[0] == "click":
        fx, fy = int(args[1]), int(args[2])
        reset(ser)
        click_abs(ser, fx, fy)

    elif args[0] == "move":
        dx, dy = int(args[1]), int(args[2])
        move_rel(ser, dx, dy)

    ser.close()
    print("\n[피코] 연결 해제")
