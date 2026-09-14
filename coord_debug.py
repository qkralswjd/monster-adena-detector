"""
coord_debug.py
--------------
좌표계 완전 디버깅 도구.

변수별 좌표계 분석:
  [A] mss_frame_xy   : mss 캡처 이미지 기준 (0~capture_w, 0~capture_h) 픽셀
  [B] det_bbox_xy    : YOLO 탐지 결과 (x1,y1,x2,y2) → [A] 와 동일 좌표계
  [C] det_cx_cy      : bbox 중심 = x1+w//2, y1+h//2 → [A] 와 동일 좌표계
  [D] screen_xy      : Windows 전체화면 좌표 = [C] + (mon_left, mon_top)
                       게임이 left=-1920이면 → D.x = C.x - 1920 (음수)
  [E] hid_move_xy    : 피코 MOVE 값 = D * SCALE (음수 그대로 전송)
                       피코 리셋(0,0)=Windows(0,0) → D음수 = 왼쪽이동 → 게임도달

모드:
  1. 정보 출력    : 해상도·DPI·모니터 레이아웃 전부 출력
  2. 좌표 추적   : 실시간 탐지 좌표 전부 출력 (클릭 안 함)
  3. 점 표시     : 캡처창에 클릭 목표점 표시 (피코 클릭 안 함)
  4. 실제 클릭   : 점 표시 + 피코 실제 클릭 실행

사용법:
  python coord_debug.py          → 모드 선택 메뉴
  python coord_debug.py --info   → 정보만 출력
  python coord_debug.py --track  → 좌표 추적만
  python coord_debug.py --dot    → 점 표시 테스트
  python coord_debug.py --click  → 실제 피코 클릭
"""

import sys
import os
import time
import json
import ctypes

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import mss


# ══════════════════════════════════════════════════════
#  유틸
# ══════════════════════════════════════════════════════

def get_windows_dpi_scale():
    """Windows DPI 배율 가져오기 (100%=1.0, 125%=1.25, 150%=1.5)"""
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0

def get_windows_cursor_pos():
    """Windows 실제 커서 위치 (전체화면 좌표)"""
    try:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    except Exception:
        return 0, 0

def get_all_monitors_info():
    """모든 모니터 정보"""
    with mss.mss() as sct:
        return sct.monitors  # [0]=전체, [1],[2]...=개별


# ══════════════════════════════════════════════════════
#  정보 출력
# ══════════════════════════════════════════════════════

def print_system_info(cfg):
    """시스템 좌표 환경 전체 출력"""
    print()
    print("=" * 65)
    print("  좌표계 환경 분석")
    print("=" * 65)

    # DPI
    dpi_scale = get_windows_dpi_scale()
    print(f"\n[DPI]")
    print(f"  Windows DPI 배율: {dpi_scale:.2f}x  ({int(dpi_scale*100)}%)")
    if dpi_scale != 1.0:
        print(f"  ⚠ DPI != 100% → mss 캡처 해상도와 실제 화면 해상도가 다를 수 있음!")
        print(f"  ⚠ ctypes.SetCursorPos 좌표도 DPI 영향 받음")

    # mss 모니터 목록
    monitors = get_all_monitors_info()
    print(f"\n[mss 모니터 목록]")
    print(f"  monitors[0] = 전체 가상 화면: "
          f"left={monitors[0]['left']} top={monitors[0]['top']} "
          f"w={monitors[0]['width']} h={monitors[0]['height']}")
    for i, m in enumerate(monitors[1:], 1):
        print(f"  monitors[{i}] = 모니터{i}: "
              f"left={m['left']:6d} top={m['top']:5d} "
              f"w={m['width']} h={m['height']}")

    # 게임 모니터
    mon_idx = cfg["capture"]["monitor"]
    mon = monitors[mon_idx]
    mon_left = mon["left"]
    mon_top  = mon["top"]
    cap_w    = mon["width"]
    cap_h    = mon["height"]
    print(f"\n[게임 모니터] monitors[{mon_idx}]")
    print(f"  left={mon_left}, top={mon_top}, width={cap_w}, height={cap_h}")
    print(f"  캡처 해상도: {cap_w} x {cap_h}")
    print(f"  config.json game.width={cfg['game']['width']} game.height={cfg['game']['height']}")
    if cap_w != cfg['game']['width'] or cap_h != cfg['game']['height']:
        print(f"  ⚠ config game 해상도와 mss 캡처 해상도 불일치!")

    # ROI
    rcfg = cfg["roi"]
    print(f"\n[ROI]")
    print(f"  프레임 기준: x={rcfg['x']} y={rcfg['y']} w={rcfg['width']} h={rcfg['height']}")
    print(f"  전체화면 기준: x={rcfg['x']+mon_left} y={rcfg['y']+mon_top}")

    # SCALE
    ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
    scale_x = scale_y = 0.395
    try:
        import re
        code = open(ctrl_path, encoding="utf-8").read()
        m = re.search(r"SCALE_X\s*=\s*([\d.]+)", code)
        if m: scale_x = float(m.group(1))
        m = re.search(r"SCALE_Y\s*=\s*([\d.]+)", code)
        if m: scale_y = float(m.group(1))
    except Exception:
        pass
    print(f"\n[HID SCALE]")
    print(f"  SCALE_X={scale_x}, SCALE_Y={scale_y}")
    print(f"  MOVE:1 → 실제 {1/scale_x:.2f}px 이동")
    print(f"  1920px 이동 시 MOVE:{int(1920*scale_x)}")

    # 좌표 변환 예시
    print(f"\n[좌표 변환 예시] 몬스터가 프레임(800, 450)에 탐지된 경우:")
    ex_fx, ex_fy = 800, 450
    ex_sx = ex_fx + mon_left
    ex_sy = ex_fy + mon_top
    ex_hx = round(ex_sx * scale_x)
    ex_hy = round(ex_sy * scale_y)
    print(f"  [A] mss 프레임   : ({ex_fx}, {ex_fy})")
    print(f"  [D] 전체화면     : ({ex_sx}, {ex_sy})  ← [A] + ({mon_left},{mon_top})")
    print(f"  [E] HID MOVE     : ({ex_hx}, {ex_hy})  ← [D] × SCALE")
    print(f"  피코 리셋 후 MOVE:{ex_hx}:{ex_hy} → 전체화면({ex_sx},{ex_sy}) 도달")

    print()
    print("=" * 65)


# ══════════════════════════════════════════════════════
#  좌표 추적 모드 (탐지만, 클릭 없음)
# ══════════════════════════════════════════════════════

def mode_track(cfg):
    """실시간 탐지 좌표 전부 출력. 피코 연결/클릭 없음."""
    from screen_capture import ScreenCapture
    from detector import YOLODetector

    import re
    ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
    code = open(ctrl_path, encoding="utf-8").read()
    scale_x = float(re.search(r"SCALE_X\s*=\s*([\d.]+)", code).group(1))
    scale_y = float(re.search(r"SCALE_Y\s*=\s*([\d.]+)", code).group(1))

    cap = ScreenCapture(cfg["capture"]["monitor"])
    mon = cap._monitor
    mon_left = mon["left"]
    mon_top  = mon["top"]
    cap_w    = mon["width"]
    cap_h    = mon["height"]

    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
        classes       = dcfg["classes"],
    )

    rcfg = cfg["roi"]
    roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"]) \
          if rcfg["width"] > 0 else None

    dpi_scale = get_windows_dpi_scale()

    print(f"\n[추적모드] 캡처={cap_w}x{cap_h}  mon=({mon_left},{mon_top})  "
          f"DPI={dpi_scale:.2f}x  SCALE=({scale_x},{scale_y})")
    print("Ctrl+C 로 종료\n")
    print(f"{'bbox(x,y,w,h)':<22} {'[A]프레임cx,cy':<16} "
          f"{'[D]전체화면x,y':<18} {'[E]HID_MOVE_x,y':<16} conf")
    print("-" * 90)

    try:
        while True:
            frame = cap.capture()
            if frame is None:
                continue
            dets = det.detect(frame)
            if roi:
                rx, ry, rw, rh = roi
                dets = [d for d in dets if rx <= d.cx <= rx+rw and ry <= d.cy <= ry+rh]
            monsters = [d for d in dets if d.class_id == 0]
            if not monsters:
                time.sleep(0.1)
                continue
            for d in monsters:
                # [A] mss 프레임 좌표
                A_cx, A_cy = d.cx, d.cy
                # [D] Windows 전체화면 좌표
                D_x = A_cx + mon_left
                D_y = A_cy + mon_top
                # [E] HID MOVE 값
                E_x = round(D_x * scale_x)
                E_y = round(D_y * scale_y)
                bbox_str = f"({d.x},{d.y},{d.w},{d.h})"
                print(f"{bbox_str:<22} ({A_cx:4d},{A_cy:4d})        "
                      f"({D_x:6d},{D_y:4d})        "
                      f"({E_x:5d},{E_y:4d})   {d.confidence:.2f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n종료")


# ══════════════════════════════════════════════════════
#  점 표시 + (선택적) 피코 클릭 모드
# ══════════════════════════════════════════════════════

def mode_dot_or_click(cfg, do_click: bool):
    """
    캡처창에 탐지된 몬스터 중심에 점 표시.
    do_click=True이면 실제 피코 클릭도 실행.

    캡처창은 원본의 절반 크기(960x540)로 표시.
    클릭 목표점: 빨간 원 + 좌표 텍스트
    """
    from screen_capture import ScreenCapture
    from detector import YOLODetector

    import re, serial as _serial
    ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
    code = open(ctrl_path, encoding="utf-8").read()
    scale_x = float(re.search(r"SCALE_X\s*=\s*([\d.]+)", code).group(1))
    scale_y = float(re.search(r"SCALE_Y\s*=\s*([\d.]+)", code).group(1))

    cap = ScreenCapture(cfg["capture"]["monitor"])
    mon = cap._monitor
    mon_left = mon["left"]
    mon_top  = mon["top"]
    cap_w    = mon["width"]
    cap_h    = mon["height"]

    dcfg = cfg["detector"]
    det = YOLODetector(
        model_path    = dcfg["model"],
        confidence    = dcfg["confidence"],
        iou_threshold = dcfg["iou_threshold"],
        device        = dcfg["device"],
        img_size      = dcfg["img_size"],
        classes       = dcfg["classes"],
    )

    rcfg = cfg["roi"]
    roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"]) \
          if rcfg["width"] > 0 else None

    # 피코 연결
    ser = None
    if do_click:
        port = cfg["controller"]["port"]
        try:
            ser = _serial.Serial(port, 115200, timeout=1)
            time.sleep(0.5)
            print(f"[피코] {port} 연결 완료")
            # 초기 리셋
            ctypes.windll.user32.SetCursorPos(0, 0)
            time.sleep(0.1)
            ser.write(b"MOVE:-9999:-9999\n"); ser.flush()
            time.sleep(1.0)
            print("[피코] 커서 리셋 완료")
        except Exception as e:
            print(f"[피코] 연결 실패: {e}")
            ser = None

    # config에서 드래그 설정 읽기
    atk = cfg.get("attack", {})
    drag_dx  = atk.get("drag_dx", 0)
    drag_dy  = atk.get("drag_dy", 30)
    hold_ms  = atk.get("hold_ms", 80)

    import threading
    _attacking = [False]  # 공격 중 플래그
    # 피코 현재 커서 위치 추적 (리셋 후 0,0 기준)
    _cur = [0, 0]  # [cur_x, cur_y] - 전체화면 좌표

    def _pico_click_thread(D_x, D_y):
        """
        별도 스레드 - 메인루프 블로킹 방지
        매번 리셋 없이 _cur 기준 상대이동 → 정확하고 빠름
        """
        _attacking[0] = True
        try:
            dE_x = round(drag_dx * scale_x)
            dE_y = round(drag_dy * scale_y)

            # 1. 현재 위치 → 목표 위치 상대이동
            dx = D_x - _cur[0]
            dy = D_y - _cur[1]
            sdx = round(dx * scale_x)
            sdy = round(dy * scale_y)
            if sdx != 0 or sdy != 0:
                ser.write(f"MOVE:{sdx}:{sdy}\n".encode()); ser.flush()
                time.sleep(0.05)
            _cur[0] = D_x
            _cur[1] = D_y

            # 2. PRESS
            ser.write(b"PRESS\n"); ser.flush()
            time.sleep(hold_ms / 1000.0)

            # 3. 드래그
            if drag_dx != 0 or drag_dy != 0:
                ser.write(f"MOVE:{dE_x}:{dE_y}\n".encode()); ser.flush()
                _cur[0] += drag_dx
                _cur[1] += drag_dy
                time.sleep(0.03)

            # 4. RELEASE
            ser.write(b"RELEASE\n"); ser.flush()
            print(f"[피코드래그] 전체화면({D_x},{D_y})  상대이동({sdx},{sdy})"
                  f"  drag({dE_x},{dE_y})  hold={hold_ms}ms")
        except Exception as e:
            print(f"[피코오류] {e}")
        finally:
            _attacking[0] = False

    def pico_click(D_x, D_y):
        """피코 드래그 공격 - 별도 스레드로 실행"""
        if ser is None:
            return
        if _attacking[0]:
            return  # 이미 공격 중이면 스킵
        t = threading.Thread(target=_pico_click_thread, args=(D_x, D_y), daemon=True)
        t.start()

    mode_name = "클릭 테스트" if do_click else "점 표시 테스트"
    WIN = f"좌표 디버그 - {mode_name} (Q=종료)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    disp_w, disp_h = cap_w // 2, cap_h // 2
    cv2.resizeWindow(WIN, disp_w, disp_h)

    last_click_time = 0
    cooldown = 2.0  # 클릭 쿨타임 (초)

    dpi_scale = get_windows_dpi_scale()
    print(f"\n[{mode_name}] DPI={dpi_scale:.2f}x  캡처={cap_w}x{cap_h}  "
          f"표시={disp_w}x{disp_h}  SCALE=({scale_x},{scale_y})")
    print("탐지된 첫 번째 몬스터에 점 표시 (빨간 원)")
    if do_click:
        print(f"  → 쿨타임 {cooldown}초마다 자동 클릭")
    print("Q / ESC = 종료\n")

    last_info = None  # 마지막 표시 정보

    try:
        while True:
            frame = cap.capture()
            if frame is None:
                continue

            dets = det.detect(frame)
            if roi:
                rx, ry, rw, rh = roi
                dets = [d for d in dets if rx <= d.cx <= rx+rw and ry <= d.cy <= ry+rh]
            monsters = [d for d in dets if d.class_id == 0]

            # 절반 크기로 축소
            disp = cv2.resize(frame, (disp_w, disp_h))

            # ROI 표시
            if roi:
                rx, ry, rw, rh = roi
                cv2.rectangle(disp,
                              (rx//2, ry//2), ((rx+rw)//2, (ry+rh)//2),
                              (0, 255, 255), 1)

            if monsters:
                m = monsters[0]  # 첫 번째 몬스터만

                # 좌표 계산
                A_cx, A_cy = m.cx, m.cy           # [A] mss 프레임
                D_x = A_cx + mon_left              # [D] Windows 전체화면
                D_y = A_cy + mon_top
                E_x = round(D_x * scale_x)         # [E] HID MOVE
                E_y = round(D_y * scale_y)

                # 디스플레이 좌표 (절반)
                dsp_cx = A_cx // 2
                dsp_cy = A_cy // 2

                # bbox 그리기 (파란색)
                bx, by, bw, bh = m.x//2, m.y//2, m.w//2, m.h//2
                cv2.rectangle(disp, (bx, by), (bx+bw, by+bh), (255, 100, 0), 2)

                # 중심점 (빨간 원) - 실제 클릭 목표
                cv2.circle(disp, (dsp_cx, dsp_cy), 8, (0, 0, 255), -1)
                cv2.circle(disp, (dsp_cx, dsp_cy), 8, (255, 255, 255), 2)
                cv2.circle(disp, (dsp_cx, dsp_cy), 20, (0, 0, 255), 2)

                # 좌표 정보 텍스트
                lines = [
                    f"[A] frame: ({A_cx},{A_cy})",
                    f"[D] screen:({D_x},{D_y})",
                    f"[E] HID:   ({E_x},{E_y})",
                    f"conf:{m.confidence:.2f}",
                ]
                ty = dsp_cy - 60
                for line in lines:
                    ty += 18
                    cv2.putText(disp, line, (dsp_cx + 12, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # 화면 하단 상세 로그
                info = (A_cx, A_cy, D_x, D_y, E_x, E_y)
                if info != last_info:
                    last_info = info
                    print(f"[탐지]"
                          f"  bbox=({m.x},{m.y},{m.w},{m.h})"
                          f"  [A]프레임=({A_cx},{A_cy})"
                          f"  [D]전체화면=({D_x},{D_y})"
                          f"  [E]HID=({E_x},{E_y})"
                          f"  conf={m.confidence:.2f}")

                # 피코 클릭
                if do_click:
                    now = time.time()
                    if now - last_click_time >= cooldown:
                        last_click_time = now
                        pico_click(D_x, D_y)
            else:
                cv2.putText(disp, "몬스터 없음", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 시스템 정보 오버레이
            info_lines = [
                f"DPI:{dpi_scale:.2f}x  cap:{cap_w}x{cap_h}",
                f"mon:({mon_left},{mon_top})  SCALE:({scale_x},{scale_y})",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(disp, line, (5, 15 + i*16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if ser:
            ser.write(b"RELEASE\n"); ser.flush()
            ser.close()
        cv2.destroyAllWindows()
        print("종료")


# ══════════════════════════════════════════════════════
#  수동 클릭 테스트 (캡처창 클릭 → 피코 클릭)
# ══════════════════════════════════════════════════════

def detect_letterbox(cfg) -> int:
    """
    게임 화면 캡처 후 좌측 검은 여백(letterbox) 너비를 자동 감지.
    검은 픽셀(R+G+B < 30) 이 끝나는 X좌표 반환.
    """
    with mss.mss() as sct:
        mon_idx = cfg["capture"]["monitor"]
        mon = sct.monitors[mon_idx]
        shot = sct.grab(mon)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w = frame.shape[:2]
    # 화면 중앙 Y라인에서 검은 픽셀 탐색
    mid_y = h // 2
    row = frame[mid_y]  # (w, 3)

    letterbox_left = 0
    for x in range(w):
        b, g, r = int(row[x][0]), int(row[x][1]), int(row[x][2])
        if r + g + b > 30:  # 검은색 아님 → 게임 시작
            letterbox_left = x
            break

    letterbox_right = w
    for x in range(w - 1, -1, -1):
        b, g, r = int(row[x][0]), int(row[x][1]), int(row[x][2])
        if r + g + b > 30:
            letterbox_right = x
            break

    game_w = letterbox_right - letterbox_left
    print(f"\n[레터박스 자동감지]")
    print(f"  왼쪽 여백: {letterbox_left}px")
    print(f"  오른쪽 여백: {w - letterbox_right}px")
    print(f"  실제 게임 영역: x={letterbox_left}~{letterbox_right} (너비 {game_w}px)")
    print(f"  → config.json letterbox_x = {letterbox_left} 으로 설정 권장")
    return letterbox_left


def mode_manual_click(cfg):
    """
    캡처창에서 마우스 클릭 → 해당 좌표 피코 클릭.
    좌표 변환 전 과정을 출력해서 어디서 틀리는지 확인.
    """
    import re, serial as _serial

    ctrl_path = os.path.join(os.path.dirname(__file__), "controller.py")
    code = open(ctrl_path, encoding="utf-8").read()
    scale_x = float(re.search(r"SCALE_X\s*=\s*([\d.]+)", code).group(1))
    scale_y = float(re.search(r"SCALE_Y\s*=\s*([\d.]+)", code).group(1))

    with mss.mss() as sct:
        mon_idx = cfg["capture"]["monitor"]
        mon = sct.monitors[mon_idx]
        mon_left = mon["left"]
        mon_top  = mon["top"]
        cap_w    = mon["width"]
        cap_h    = mon["height"]

    dpi_scale = get_windows_dpi_scale()
    disp_w, disp_h = cap_w // 2, cap_h // 2

    port = cfg["controller"]["port"]
    ser = None
    # 피코 현재 커서 위치 추적 (전체화면 좌표, 리셋 후 0,0 기준)
    pico_cur = [0, 0]
    try:
        ser = _serial.Serial(port, 115200, timeout=1)
        time.sleep(0.5)
        print(f"[피코] {port} 연결 완료")
        ctypes.windll.user32.SetCursorPos(0, 0)
        time.sleep(0.1)
        ser.write(b"MOVE:-9999:-9999\n"); ser.flush()
        time.sleep(1.5)  # 리셋 완료 충분히 대기
        pico_cur[0] = 0
        pico_cur[1] = 0
        print("[피코] 커서 리셋 완료 → pico(0,0) = Windows(0,0)")
    except Exception as e:
        print(f"[피코] 연결 실패: {e}  → 좌표 계산만 출력")

    click_queue = []
    import threading
    lock = threading.Lock()

    # 실제 창 이미지 영역 크기 (cv2.getWindowImageRect로 정확히 읽음)
    # 초기값은 disp_w/h, 루프에서 매 프레임 갱신
    win_img_rect = [0, 0, disp_w, disp_h]  # [x, y, w, h]

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 실제 이미지 영역 크기로 비율 계산
            ww = win_img_rect[2] if win_img_rect[2] > 0 else disp_w
            wh = win_img_rect[3] if win_img_rect[3] > 0 else disp_h
            ratio_x = cap_w / ww
            ratio_y = cap_h / wh
            # [A] 프레임 좌표 = 클릭좌표 × 실제비율
            A_x = round(mx * ratio_x)
            A_y = round(my * ratio_y)
            # [D] 전체화면 좌표
            D_x = A_x + mon_left
            D_y = A_y + mon_top
            # [E] HID MOVE
            E_x = round(D_x * scale_x)
            E_y = round(D_y * scale_y)
            with lock:
                click_queue.append((mx, my, A_x, A_y, D_x, D_y, E_x, E_y,
                                    ww, wh, ratio_x, ratio_y))

    WIN = "수동 클릭 테스트 (Q=종료)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, on_mouse)

    print(f"\n[수동클릭] DPI={dpi_scale:.2f}x  캡처={cap_w}x{cap_h}  "
          f"mon=({mon_left},{mon_top})  SCALE=({scale_x},{scale_y})")
    print("캡처창 아무 곳이나 클릭 → 좌표 출력 + 피코 클릭")
    print("Q / ESC = 종료\n")
    print(f"{'클릭(disp)':<14} {'[A]프레임':<14} {'[D]전체화면':<16} {'[E]HID_MOVE'}")
    print("-" * 65)

    sct2 = mss.mss()
    mon_rect = sct2.monitors[cfg["capture"]["monitor"]]
    last_dot = None

    try:
        while True:
            # ── win_img_rect 갱신: 실제 이미지 영역 크기 (타이틀바 제외) ──
            try:
                r = cv2.getWindowImageRect(WIN)
                if r[2] > 0 and r[3] > 0:
                    win_img_rect[0] = r[0]
                    win_img_rect[1] = r[1]
                    win_img_rect[2] = r[2]
                    win_img_rect[3] = r[3]
            except Exception:
                pass

            shot = sct2.grab(mon_rect)
            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            disp = cv2.resize(frame, (disp_w, disp_h))

            with lock:
                if click_queue:
                    mx, my, A_x, A_y, D_x, D_y, E_x, E_y, ww, wh, ratio_x, ratio_y = click_queue.pop(0)

                    print(f"({mx:4d},{my:4d})  "
                          f"imgRect({ww}x{wh})  ratio({ratio_x:.3f},{ratio_y:.3f})")
                    print(f"  → [A]프레임({A_x:4d},{A_y:4d})  "
                          f"[D]전체화면({D_x:6d},{D_y:4d})  "
                          f"[E]HID({E_x:5d},{E_y:4d})")

                    # 피코 클릭 - 상대이동 방식
                    if ser:
                        rdx = round((D_x - pico_cur[0]) * scale_x)
                        rdy = round((D_y - pico_cur[1]) * scale_y)
                        if rdx != 0 or rdy != 0:
                            ser.write(f"MOVE:{rdx}:{rdy}\n".encode()); ser.flush()
                            time.sleep(0.05)
                        pico_cur[0] = D_x
                        pico_cur[1] = D_y
                        ser.write(b"CLICK:80\n"); ser.flush()
                        print(f"  → 피코 클릭: 상대이동({rdx},{rdy})  목표전체화면({D_x},{D_y})")

                    last_dot = (mx, my, D_x, D_y)

            # 마지막 클릭 점 표시
            if last_dot:
                dx, dy, scx, scy = last_dot
                cv2.circle(disp, (dx, dy), 10, (0, 0, 255), -1)
                cv2.circle(disp, (dx, dy), 10, (255, 255, 255), 2)
                cv2.circle(disp, (dx, dy), 22, (0, 0, 255), 2)
                cv2.putText(disp, f"screen({scx},{scy})",
                            (dx+14, dy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 오버레이
            cv2.putText(disp, f"DPI:{dpi_scale:.2f}x  cap:{cap_w}x{cap_h}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)
            cv2.putText(disp, f"mon:({mon_left},{mon_top}) SCALE:({scale_x},{scale_y})",
                        (5, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if ser:
            ser.write(b"RELEASE\n"); ser.flush()
            ser.close()
        sct2.close()
        cv2.destroyAllWindows()
        print("종료")


# ══════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    args = sys.argv[1:]

    if "--letterbox" in args:
        detect_letterbox(cfg)
        return

    if "--info" in args:
        print_system_info(cfg)
        return

    if "--track" in args:
        print_system_info(cfg)
        mode_track(cfg)
        return

    if "--dot" in args:
        print_system_info(cfg)
        mode_dot_or_click(cfg, do_click=False)
        return

    if "--click" in args:
        print_system_info(cfg)
        mode_dot_or_click(cfg, do_click=True)
        return

    if "--manual" in args:
        print_system_info(cfg)
        mode_manual_click(cfg)
        return

    # 메뉴
    print_system_info(cfg)
    print("\n모드 선택:")
    print("  1. 좌표 추적만 (탐지 좌표 실시간 출력, 클릭 없음)")
    print("  2. 점 표시 테스트 (캡처창에 클릭 목표점 표시, 피코 클릭 없음)")
    print("  3. 자동 클릭 테스트 (점 표시 + 피코 실제 클릭)")
    print("  4. 수동 클릭 테스트 (캡처창 클릭 → 피코 클릭)")
    print("  q. 종료")
    choice = input("\n선택 > ").strip()

    if choice == "1":
        mode_track(cfg)
    elif choice == "2":
        mode_dot_or_click(cfg, do_click=False)
    elif choice == "3":
        mode_dot_or_click(cfg, do_click=True)
    elif choice == "4":
        mode_manual_click(cfg)
    else:
        print("종료")


if __name__ == "__main__":
    main()
