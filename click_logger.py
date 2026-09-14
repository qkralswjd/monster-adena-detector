"""
click_logger.py
---------------
ctypes Win32 글로벌 마우스 훅으로 클릭 좌표 출력.
게임(DirectInput/Raw Input) 환경에서도 동작.

실행: python click_logger.py
종료: Ctrl+C
"""

import ctypes
import ctypes.wintypes
import sys
import os
import json
import time

# ── 게임 모니터 정보 로드 ──────────────────────────────────────
cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

import mss
with mss.MSS() as sct:
    mon_idx = cfg["capture"]["monitor"]
    mon = sct.monitors[mon_idx]

MON_LEFT   = mon["left"]
MON_TOP    = mon["top"]
MON_WIDTH  = mon["width"]
MON_HEIGHT = mon["height"]

print("=" * 60)
print(f"  클릭 좌표 로거 (Win32 훅)")
print(f"  게임 모니터: left={MON_LEFT}, top={MON_TOP}")
print(f"  캡처 크기:   {MON_WIDTH} x {MON_HEIGHT}")
print(f"  게임 화면 아무 곳이나 클릭하면 좌표 출력")
print(f"  종료: Ctrl+C")
print("=" * 60)

# ── Win32 상수 ────────────────────────────────────────────────
WH_MOUSE_LL   = 14
WM_LBUTTONDOWN = 0x0201

click_count = 0

# ── MSLLHOOKSTRUCT ────────────────────────────────────────────
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt",      POINT),
        ("mouseData", ctypes.wintypes.DWORD),
        ("flags",     ctypes.wintypes.DWORD),
        ("time",      ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

# ── 훅 콜백 ──────────────────────────────────────────────────
HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM
)

def low_level_mouse_proc(nCode, wParam, lParam):
    global click_count
    if nCode >= 0 and wParam == WM_LBUTTONDOWN:
        ms = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
        abs_x = ms.pt.x
        abs_y = ms.pt.y

        frame_x = abs_x - MON_LEFT
        frame_y = abs_y - MON_TOP

        click_count += 1

        in_game = (0 <= frame_x <= MON_WIDTH and 0 <= frame_y <= MON_HEIGHT)
        zone = "게임내" if in_game else "게임밖"

        ratio_x = frame_x / MON_WIDTH
        ratio_y = frame_y / MON_HEIGHT

        print(f"[클릭 #{click_count:03d}][{zone}] "
              f"프레임({frame_x:5d},{frame_y:5d})  "
              f"화면비=({ratio_x:.3f},{ratio_y:.3f})  "
              f"절대=({abs_x},{abs_y})")

    return ctypes.windll.user32.CallNextHookEx(None, nCode, wParam, lParam)

# ── 훅 설치 및 메시지 루프 ────────────────────────────────────
hook_proc   = HOOKPROC(low_level_mouse_proc)
hook_handle = ctypes.windll.user32.SetWindowsHookExA(
    WH_MOUSE_LL, hook_proc, None, 0
)

if not hook_handle:
    print("❌ 훅 설치 실패 (관리자 권한으로 실행해보세요)")
    sys.exit(1)

print("✅ 훅 설치 완료. 클릭하세요...\n")

msg = ctypes.wintypes.MSG()
try:
    while ctypes.windll.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
        ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
        ctypes.windll.user32.DispatchMessageA(ctypes.byref(msg))
except KeyboardInterrupt:
    pass
finally:
    ctypes.windll.user32.UnhookWindowsHookEx(hook_handle)
    print(f"\n종료. 총 {click_count}번 클릭 기록")
