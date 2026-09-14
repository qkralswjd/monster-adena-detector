"""
test_drag.py
------------
피코 PRESS/MOVE/RELEASE 드래그 단계별 테스트.

1단계: PING/PONG 연결 확인
2단계: 단순 CLICK 테스트
3단계: PRESS → RELEASE (드래그 없이 누르고 떼기)
4단계: PRESS → MOVE → RELEASE (실제 드래그)

실행:
    python test_drag.py
"""

import serial
import time

PORT = "COM4"
BAUD = 115200

print(f"[Test] {PORT} 연결 중...")
s = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(0.5)
print(f"[Test] 연결 완료\n")


def send(cmd: str):
    data = (cmd + "\n").encode("utf-8")
    s.write(data)
    s.flush()
    print(f"  >> 전송: {cmd}")


def recv():
    resp = s.readline().decode("utf-8", "ignore").strip()
    print(f"  << 수신: {resp!r}")
    return resp


def wait(sec: float, msg: str = ""):
    if msg:
        print(f"  ... {msg} ({sec}s 대기)")
    time.sleep(sec)


# ══════════════════════════════════════════════
print("=" * 50)
print("1단계: PING/PONG 연결 확인")
print("=" * 50)
send("PING")
resp = recv()
if resp == "PONG":
    print("  ✅ PONG 정상 수신")
else:
    print(f"  ⚠ 예상: PONG, 실제: {resp!r}")
wait(0.5)

# ══════════════════════════════════════════════
print()
print("=" * 50)
print("2단계: 단순 CLICK 테스트 (5초 후 클릭)")
print("       마우스 커서를 아무 곳에 놓고 클릭되는지 확인")
print("=" * 50)
wait(5, "클릭 준비")
send("CLICK:100")
wait(0.5)
print("  → 클릭이 됐나요? (y/n)")

# ══════════════════════════════════════════════
print()
print("=" * 50)
print("3단계: PRESS → 1초 유지 → RELEASE")
print("       버튼이 눌리고 떼지는지 확인 (드래그 없음)")
print("=" * 50)
wait(3, "준비")
send("PRESS")
wait(1.0, "1초 누름 유지")
send("RELEASE")
wait(0.5)
print("  → PRESS/RELEASE 됐나요? (y/n)")

# ══════════════════════════════════════════════
print()
print("=" * 50)
print("4단계: PRESS → MOVE:50:0 → RELEASE (오른쪽 드래그)")
print("       마우스를 누른 채로 오른쪽으로 드래그되는지 확인")
print("=" * 50)
wait(3, "준비")
send("PRESS")
wait(0.1)
send("MOVE:50:0")
wait(0.2)
send("RELEASE")
wait(0.5)
print("  → 드래그 됐나요? (y/n)")

# ══════════════════════════════════════════════
print()
print("=" * 50)
print("5단계: STOP (버튼 강제 해제)")
print("=" * 50)
send("STOP")
wait(0.3)

print()
print("=== 테스트 완료 ===")
print("각 단계 결과를 알려주세요!")
s.close()
