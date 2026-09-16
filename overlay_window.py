"""
overlay_window.py
-----------------
게임화면 위에 올라가는 투명 오버레이 창.

특징:
  - 항상 게임화면 위에 표시 (topmost)
  - 배경 완전 투명 → 게임화면이 비쳐 보임
  - 탐지된 몬스터 박스 / 원 / 텍스트 표시
  - ROI 영역을 시안색 점선 박스로 표시 (전체화면 기준)

설정 모드 (한 번만 실행):
  F1  → 허수아비 좌표 찍기 모드  (우클릭으로 고정좌표 등록)
  F2  → 웨이포인트 찍기 모드     (우클릭으로 최대 10개 등록)
  F3  → 레벨 UI ROI 지정 모드   (드래그로 영역 지정)
  F4  → HP바 ROI 지정 모드      (드래그로 영역 지정)
  ESC → 설정 모드 종료 / 일반 모드로 복귀

설정값은 config.json 에 자동 저장됨.
"""

import tkinter as tk
import threading
import time
import json
import os
import math
import ctypes
import ctypes.wintypes
from typing import List, Optional, Callable

# Windows 스타일 상수
GWL_EXSTYLE     = -20
WS_EX_LAYERED   = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

def _get_hwnd(root):
    """tkinter 창의 HWND 반환."""
    return ctypes.windll.user32.FindWindowW(None, root.title() or None)

def _set_click_through(hwnd, enable: bool):
    """
    enable=True  → 클릭 통과 (게임으로 전달, 오버레이 표시 전용)
    enable=False → 클릭 수신 (설정 모드, 오버레이가 클릭 받음)
    """
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if enable:
        style |= WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


# ── 색상 ────────────────────────────────────────────────────
CLR_BOX       = "#FF6600"
CLR_TARGET    = "#00FF44"
CLR_DEAD      = "#FF2222"
CLR_ADENA     = "#FFD700"
CLR_ADENA_TGT = "#FF44FF"
CLR_HUD_BG    = "#000000"
CLR_TEXT      = "#FFFFFF"
CLR_TARGET_TXT= "#00FF44"
CLR_ROI       = "#00FFFF"

# 설정모드 색상
CLR_DUMMY     = "#FF4444"   # 허수아비 (빨강)
CLR_WAYPOINT  = "#44FF44"   # 웨이포인트 (초록)
CLR_LEVEL_ROI = "#FFFF00"   # 레벨 UI ROI (노랑)
CLR_HP_ROI    = "#FF6666"   # HP바 ROI (연빨강)
CLR_SETUP_BG  = "#222222"   # 설정모드 HUD 배경

TRANSPARENT_KEY = "#010101"

# 설정 모드
MODE_NORMAL      = "NORMAL"
MODE_DUMMY       = "SETUP_DUMMY"
MODE_WAYPOINT    = "SETUP_WAYPOINT"
MODE_LEVEL_ROI   = "SETUP_LEVEL_ROI"
MODE_HP_ROI      = "SETUP_HP_ROI"

MAX_WAYPOINTS = 10


class OverlayWindow:
    """
    게임화면 위 투명 오버레이.

    좌표계:
        오버레이 창 = 전체 모니터 크기 (mon_left, mon_top) 에서 (mon_w x mon_h)
        탐지 좌표 (det.cx, det.cy) = ROI 캡처 기준 → roi_x/roi_y 오프셋 적용
    """

    def __init__(self,
                 mon_left: int,
                 mon_top: int,
                 game_w: int,
                 game_h: int,
                 lb_x: int,
                 ctrl=None,
                 roi: dict = None,
                 mon_w: int = 1920,
                 mon_h: int = 1080,
                 config_path: str = None,
                 on_config_saved: Callable = None):
        """
        mon_left, mon_top  : mss 모니터의 left/top
        game_w, game_h     : 캡처 영역 크기
        lb_x               : 좌측 letterbox 너비
        ctrl               : PicoController 또는 DummyController
        roi                : config.json 의 roi 딕셔너리
        mon_w, mon_h       : 전체 모니터 해상도
        config_path        : config.json 경로 (설정 저장용)
        on_config_saved    : 설정 저장 후 콜백 (main.py에서 reload 등)
        """
        self._mon_left = mon_left
        self._mon_top  = mon_top
        self._game_w   = game_w
        self._game_h   = game_h
        self._lb_x     = lb_x
        self._ctrl     = ctrl
        self._mon_w    = mon_w
        self._mon_h    = mon_h
        self._config_path     = config_path
        self._on_config_saved = on_config_saved

        # ROI 설정
        self._roi = None
        if roi and roi.get("enabled", False):
            self._roi = roi

        # 오버레이 창 = 전체 모니터 크기
        self._win_x = mon_left
        self._win_y = mon_top
        self._ov_w  = mon_w
        self._ov_h  = mon_h

        # ── 공유 데이터 (메인스레드 → tkinter) ──────────────
        self._lock       = threading.Lock()
        self._detections = []
        self._target     = None
        self._miss_t     = 0.0
        self._det_fps    = 0.0
        self._cap_fps    = 0.0
        self._state      = "HUNTING"
        self._hp_pct     = None   # 0.0~1.0 or None
        self._level      = None   # int or None

        self._running    = False
        self._root       = None
        self._canvas     = None
        self._thread     = None

        # ── 설정 모드 상태 ───────────────────────────────────
        self._setup_mode   = MODE_NORMAL
        self._dummy_pos    = None          # (x, y) 절대좌표
        self._waypoints    = []            # [(x,y), ...]
        self._level_roi    = None          # {"x","y","w","h"}
        self._hp_roi       = None          # {"x","y","w","h"}

        # 드래그 상태 (레벨/HP ROI 지정용)
        self._drag_start   = None          # (x, y)
        self._drag_current = None          # (x, y)
        self._dragging     = False

        # 클릭 쿨다운
        self._last_click   = 0.0
        self._click_cooldown = 0.3

        # 설정 저장 콜백용 플래그
        self._config_dirty = False

    # ─────────────────────────────────────────────────────────
    #  외부 API
    # ─────────────────────────────────────────────────────────

    def update(self, detections, target=None, miss_elapsed=0.0,
               det_fps=0.0, cap_fps=0.0, state="HUNTING",
               hp_pct=None, level=None):
        """메인루프에서 매 프레임 호출."""
        with self._lock:
            self._detections = list(detections)
            self._target     = target
            self._miss_t     = miss_elapsed
            self._det_fps    = det_fps
            self._cap_fps    = cap_fps
            self._state      = state
            self._hp_pct     = hp_pct
            self._level      = level

    def get_setup_data(self) -> dict:
        """현재 설정값 반환 (main.py에서 config 저장용)."""
        return {
            "dummy_pos":  self._dummy_pos,
            "waypoints":  list(self._waypoints),
            "level_roi":  self._level_roi,
            "hp_roi":     self._hp_roi,
        }

    def load_setup(self, dummy_pos=None, waypoints=None,
                   level_roi=None, hp_roi=None):
        """config.json 에서 불러온 설정값 적용."""
        if dummy_pos:
            self._dummy_pos = tuple(dummy_pos)
        if waypoints:
            self._waypoints = [tuple(w) for w in waypoints]
        if level_roi:
            self._level_roi = level_roi
        if hp_roi:
            self._hp_roi = hp_roi

    def start(self):
        """별도 스레드에서 tkinter 루프 시작."""
        self._running = True
        self._thread  = threading.Thread(target=self._tk_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._root:
            try:
                self._root.quit()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────
    #  tkinter 루프
    # ─────────────────────────────────────────────────────────

    def _tk_loop(self):
        self._root = tk.Tk()
        root = self._root

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", TRANSPARENT_KEY)
        root.attributes("-alpha", 1.0)
        root.configure(bg=TRANSPARENT_KEY)
        root.geometry(f"{self._ov_w}x{self._ov_h}+{self._win_x}+{self._win_y}")

        self._canvas = tk.Canvas(
            root,
            width=self._ov_w,
            height=self._ov_h,
            bg=TRANSPARENT_KEY,
            highlightthickness=0
        )
        self._canvas.pack()

        # ── HWND 저장 (클릭통과 전환용) ──────────────────────
        root.update()  # 창 생성 완료 대기
        try:
            hwnd_child = root.winfo_id()
            # GA_ROOT=2: 최상위 부모 창 HWND 획득
            self._hwnd = ctypes.windll.user32.GetAncestor(hwnd_child, 2)
        except Exception:
            self._hwnd = None

        # 시작 시 클릭 통과 ON (일반 모드)
        if self._hwnd:
            _set_click_through(self._hwnd, True)

        # ── 바인딩 ────────────────────────────────────────────
        # 일반 클릭 (좌클릭) → 게임 입력
        self._canvas.bind("<Button-1>",         self._on_left_click)
        # 우클릭 → 좌표 등록 (설정 모드)
        self._canvas.bind("<Button-3>",         self._on_right_click)
        # 드래그 (ROI 지정)
        self._canvas.bind("<ButtonPress-1>",    self._on_drag_start)
        self._canvas.bind("<B1-Motion>",        self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>",  self._on_drag_end)

        # ── 전역 핫키 (pynput) ───────────────────────────────
        # tkinter 포커스 없어도 작동
        self._start_hotkey_listener()

        self._schedule_redraw()
        root.mainloop()

    def _start_hotkey_listener(self):
        """pynput 전역 키보드 리스너 시작 (별도 스레드)."""
        try:
            from pynput import keyboard as _kb

            _ctrl_pressed = [False]

            def on_press(key):
                try:
                    if key == _kb.Key.ctrl_l or key == _kb.Key.ctrl_r:
                        _ctrl_pressed[0] = True
                    elif key == _kb.Key.f1:
                        self._root.after(0, lambda: self._set_mode(MODE_DUMMY))
                    elif key == _kb.Key.f2:
                        self._root.after(0, lambda: self._set_mode(MODE_WAYPOINT))
                    elif key == _kb.Key.f3:
                        self._root.after(0, lambda: self._set_mode(MODE_LEVEL_ROI))
                    elif key == _kb.Key.f4:
                        self._root.after(0, lambda: self._set_mode(MODE_HP_ROI))
                    elif key == _kb.Key.esc:
                        self._root.after(0, lambda: self._set_mode(MODE_NORMAL))
                    elif key == _kb.Key.delete:
                        self._root.after(0, self._undo_last)
                    elif _ctrl_pressed[0]:
                        try:
                            if key.char == 'x' or key.char == 'X':
                                self._root.after(0, self._save_config)
                        except AttributeError:
                            pass
                except Exception:
                    pass

            def on_release(key):
                try:
                    if key == _kb.Key.ctrl_l or key == _kb.Key.ctrl_r:
                        _ctrl_pressed[0] = False
                except Exception:
                    pass

            listener = _kb.Listener(on_press=on_press, on_release=on_release)
            listener.daemon = True
            listener.start()
            print("[Overlay] 전역 핫키 리스너 시작 (pynput)")
            print("[Overlay] F1/F2/F3/F4: 설정모드  ESC: 종료  Ctrl+X: 저장  Delete: 되돌리기")
        except ImportError:
            print("[Overlay] pynput 없음 → pip install pynput")
        except Exception as e:
            print(f"[Overlay] 핫키 리스너 실패: {e}")

    def _set_mode(self, mode: str):
        self._setup_mode = mode
        self._drag_start   = None
        self._drag_current = None
        self._dragging     = False
        print(f"[Setup] 모드 전환: {mode}")

        # 클릭 통과 전환
        # 설정 모드 → 클릭 수신 (오버레이가 마우스 받음)
        # 일반 모드 → 클릭 통과 (게임으로 전달)
        if self._hwnd:
            click_through = (mode == MODE_NORMAL)
            _set_click_through(self._hwnd, click_through)
            print(f"[Setup] 클릭통과: {'ON' if click_through else 'OFF (오버레이가 클릭 수신)'}")

        if mode == MODE_NORMAL:
            print("[Setup] 일반 모드 (설정 완료)")

    def _schedule_redraw(self):
        if not self._running:
            return
        self._redraw()
        if self._root:
            self._root.after(33, self._schedule_redraw)

    # ─────────────────────────────────────────────────────────
    #  마우스 이벤트
    # ─────────────────────────────────────────────────────────

    def _on_left_click(self, event):
        """일반 모드: 좌클릭 → 피코 클릭. 설정 모드: 드래그 시작."""
        if self._setup_mode != MODE_NORMAL:
            return  # 드래그 핸들러가 처리
        now = time.time()
        if now - self._last_click < self._click_cooldown:
            return
        self._last_click = now
        sc_x = self._win_x + event.x
        sc_y = self._win_y + event.y
        if self._ctrl and self._ctrl.is_connected:
            self._ctrl.drag_attack(sc_x, sc_y)

    def _on_right_click(self, event):
        """우클릭 → 설정 모드에서 좌표 등록."""
        sc_x = self._win_x + event.x
        sc_y = self._win_y + event.y

        if self._setup_mode == MODE_DUMMY:
            self._dummy_pos = (sc_x, sc_y)
            print(f"[Setup] 허수아비 좌표 등록: ({sc_x}, {sc_y})")

        elif self._setup_mode == MODE_WAYPOINT:
            if len(self._waypoints) < MAX_WAYPOINTS:
                self._waypoints.append((sc_x, sc_y))
                print(f"[Setup] 웨이포인트 {len(self._waypoints)} 등록: ({sc_x}, {sc_y})")
            else:
                print(f"[Setup] 웨이포인트 최대 {MAX_WAYPOINTS}개 초과")

    def _on_drag_start(self, event):
        """드래그 시작 (ROI 설정 모드에서만)."""
        if self._setup_mode not in (MODE_LEVEL_ROI, MODE_HP_ROI):
            return
        self._drag_start   = (event.x, event.y)
        self._drag_current = (event.x, event.y)
        self._dragging     = True

    def _on_drag_move(self, event):
        """드래그 중."""
        if not self._dragging:
            return
        self._drag_current = (event.x, event.y)

    def _on_drag_end(self, event):
        """드래그 종료 → ROI 확정."""
        if not self._dragging or self._drag_start is None:
            return
        self._dragging = False

        x1, y1 = self._drag_start
        x2, y2 = event.x, event.y

        # 정규화 (좌상→우하)
        rx = min(x1, x2) + self._win_x
        ry = min(y1, y2) + self._win_y
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        if rw < 5 or rh < 5:
            print("[Setup] 드래그 영역 너무 작음 → 무시")
            self._drag_start   = None
            self._drag_current = None
            return

        roi = {"x": rx, "y": ry, "w": rw, "h": rh}

        if self._setup_mode == MODE_LEVEL_ROI:
            self._level_roi = roi
            print(f"[Setup] 레벨 UI ROI 등록: {roi}")
        elif self._setup_mode == MODE_HP_ROI:
            self._hp_roi = roi
            print(f"[Setup] HP바 ROI 등록: {roi}")

        self._drag_start   = None
        self._drag_current = None

    def _undo_last(self):
        """Delete키 → 마지막 등록 취소."""
        if self._setup_mode == MODE_WAYPOINT and self._waypoints:
            removed = self._waypoints.pop()
            print(f"[Setup] 웨이포인트 마지막 삭제: {removed}  남은={len(self._waypoints)}")
        elif self._setup_mode == MODE_DUMMY:
            self._dummy_pos = None
            print("[Setup] 허수아비 좌표 초기화")
        elif self._setup_mode == MODE_LEVEL_ROI:
            self._level_roi = None
            print("[Setup] 레벨 UI ROI 초기화")
        elif self._setup_mode == MODE_HP_ROI:
            self._hp_roi = None
            print("[Setup] HP바 ROI 초기화")

    def _save_config(self):
        """Enter키 → config.json 저장."""
        if not self._config_path:
            print("[Setup] config_path 미설정 → 저장 불가")
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = json.load(f)

            # dummy 섹션
            if self._dummy_pos:
                cfg.setdefault("dummy", {})["pos"] = list(self._dummy_pos)

            # waypoints 섹션
            if self._waypoints:
                cfg.setdefault("movement", {})["waypoints"] = [
                    list(w) for w in self._waypoints
                ]

            # level_roi 섹션
            if self._level_roi:
                cfg.setdefault("level_detector", {})["level_roi"] = self._level_roi

            # hp_roi 섹션
            if self._hp_roi:
                cfg.setdefault("level_detector", {})["hp_roi"] = self._hp_roi

            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            print(f"[Setup] config.json 저장 완료")
            print(f"  허수아비: {self._dummy_pos}")
            print(f"  웨이포인트: {len(self._waypoints)}개")
            print(f"  레벨ROI: {self._level_roi}")
            print(f"  HPROI: {self._hp_roi}")

            if self._on_config_saved:
                self._on_config_saved()

        except Exception as e:
            print(f"[Setup] 저장 실패: {e}")

    # ─────────────────────────────────────────────────────────
    #  좌표 변환
    # ─────────────────────────────────────────────────────────

    def _det_to_ov(self, dx, dy):
        """탐지 좌표(ROI 기준) → 오버레이 좌표(전체 모니터 기준)."""
        if self._roi:
            return dx + self._roi["x"], dy + self._roi["y"]
        return dx - self._lb_x, dy

    # ─────────────────────────────────────────────────────────
    #  그리기
    # ─────────────────────────────────────────────────────────

    def _redraw(self):
        c = self._canvas
        if c is None:
            return
        c.delete("all")

        with self._lock:
            dets    = list(self._detections)
            target  = self._target
            miss_t  = self._miss_t
            det_fps = self._det_fps
            cap_fps = self._cap_fps
            state   = self._state
            hp_pct  = self._hp_pct
            level   = self._level

        # ── ROI 박스 ──────────────────────────────────────────
        if self._roi:
            rx  = self._roi["x"]
            ry  = self._roi["y"]
            rw  = self._roi["width"]
            rh  = self._roi["height"]
            rx2 = rx + rw
            ry2 = ry + rh
            if ry > 0:
                c.create_rectangle(0, 0, self._ov_w, ry,
                                   fill="#000000", outline="", stipple="gray25")
            if ry2 < self._ov_h:
                c.create_rectangle(0, ry2, self._ov_w, self._ov_h,
                                   fill="#000000", outline="", stipple="gray25")
            if rx > 0:
                c.create_rectangle(0, ry, rx, ry2,
                                   fill="#000000", outline="", stipple="gray25")
            if rx2 < self._ov_w:
                c.create_rectangle(rx2, ry, self._ov_w, ry2,
                                   fill="#000000", outline="", stipple="gray25")
            c.create_rectangle(rx, ry, rx2, ry2,
                               outline=CLR_ROI, width=2, dash=(8, 4))
            self._text(c, f"ROI  {rw}x{rh}  ({rx},{ry})",
                       rx + 4, ry + 16, CLR_ROI, size=9)

        # ── 탐지 박스 ─────────────────────────────────────────
        for d in dets:
            is_tgt = (target is not None and
                      d.x == target.x and d.y == target.y)
            ox,  oy  = self._det_to_ov(d.x,  d.y)
            ox2, oy2 = self._det_to_ov(d.x + d.w, d.y + d.h)
            ocx, ocy = self._det_to_ov(d.cx, d.cy)
            if is_tgt:
                continue
            if d.class_id == 1:
                adena_clr = CLR_ADENA_TGT if state == "ADENA_CHECK" else CLR_ADENA
                adena_w   = 2             if state == "ADENA_CHECK" else 1
                c.create_rectangle(ox, oy, ox2, oy2,
                                   outline=adena_clr, width=adena_w)
                self._text(c, f"adena {d.confidence:.2f}",
                           ox, oy - 4, adena_clr, size=9)
            else:
                c.create_rectangle(ox, oy, ox2, oy2,
                                   outline=CLR_BOX, width=1)
                self._text(c, f"{d.confidence:.2f}",
                           ox, oy - 4, CLR_BOX, size=9)
                c.create_oval(ocx-3, ocy-3, ocx+3, ocy+3,
                              fill=CLR_BOX, outline="")

        # ── 타겟 박스 ─────────────────────────────────────────
        if target is not None:
            ox,  oy  = self._det_to_ov(target.x,            target.y)
            ox2, oy2 = self._det_to_ov(target.x + target.w, target.y + target.h)
            ocx, ocy = self._det_to_ov(target.cx, target.cy)
            color = CLR_DEAD if miss_t > 0 else CLR_TARGET
            label = (f"MISSING {miss_t:.1f}s" if miss_t > 0
                     else f"TARGET  {target.confidence:.2f}")
            thick = 1 if miss_t > 0 else 2
            c.create_rectangle(ox, oy, ox2, oy2, outline=color, width=thick)
            cs = 14
            for px, py, ddx, ddy in [
                (ox,  oy,   cs,  cs),
                (ox2, oy,  -cs,  cs),
                (ox,  oy2,  cs, -cs),
                (ox2, oy2, -cs, -cs),
            ]:
                c.create_line(px, py, px+ddx, py, fill=color, width=3)
                c.create_line(px, py, px, py+ddy, fill=color, width=3)
            r = 6
            c.create_oval(ocx-r, ocy-r, ocx+r, ocy+r, fill=color, outline="")
            self._text(c, label, ox, oy - 10, color, size=10)

        # ── 설정 모드 오버레이 ────────────────────────────────
        if self._setup_mode != MODE_NORMAL:
            self._draw_setup_overlay(c)
        else:
            self._draw_registered_markers(c)

        # ── 메인 HUD (좌상단) ─────────────────────────────────
        self._draw_hud(c, dets, target, det_fps, cap_fps, state, hp_pct, level)

    def _draw_setup_overlay(self, c):
        """설정 모드일 때 안내 + 등록된 좌표 표시."""

        # 반투명 상단 배너
        c.create_rectangle(0, 0, self._ov_w, 50,
                           fill="#000000", outline="", stipple="gray75")

        mode_labels = {
            MODE_DUMMY:     ("F1  허수아비 좌표 설정", CLR_DUMMY,    "우클릭: 허수아비 위치 고정  |  Delete: 초기화  |  Enter: 저장  |  ESC: 종료"),
            MODE_WAYPOINT:  ("F2  웨이포인트 설정",    CLR_WAYPOINT, f"우클릭: 웨이포인트 추가 ({len(self._waypoints)}/{MAX_WAYPOINTS})  |  Delete: 마지막 삭제  |  Enter: 저장  |  ESC: 종료"),
            MODE_LEVEL_ROI: ("F3  레벨 UI ROI 설정",  CLR_LEVEL_ROI,"드래그: 레벨 표시 영역 지정  |  Delete: 초기화  |  Enter: 저장  |  ESC: 종료"),
            MODE_HP_ROI:    ("F4  HP바 ROI 설정",     CLR_HP_ROI,   "드래그: HP바 영역 지정  |  Delete: 초기화  |  Enter: 저장  |  ESC: 종료"),
        }
        title, clr, hint = mode_labels.get(self._setup_mode, ("", CLR_TEXT, ""))
        self._text(c, f"[ 설정 모드 ]  {title}", 10, 22, clr, size=13)
        self._text(c, hint, 10, 42, "#CCCCCC", size=9)

        # ── 허수아비 좌표 표시 ────────────────────────────────
        if self._dummy_pos:
            dx, dy = self._dummy_pos
            r = 18
            c.create_oval(dx-r, dy-r, dx+r, dy+r,
                          outline=CLR_DUMMY, width=3)
            c.create_line(dx-r-5, dy, dx+r+5, dy, fill=CLR_DUMMY, width=2)
            c.create_line(dx, dy-r-5, dx, dy+r+5, fill=CLR_DUMMY, width=2)
            self._text(c, f"허수아비 ({dx},{dy})", dx+r+4, dy+6, CLR_DUMMY, size=10)

        # ── 웨이포인트 표시 ───────────────────────────────────
        prev = None
        for i, (wx, wy) in enumerate(self._waypoints):
            # 라인 연결
            if prev:
                c.create_line(prev[0], prev[1], wx, wy,
                              fill=CLR_WAYPOINT, width=2, dash=(6, 3))
            # 마커
            r = 10
            c.create_oval(wx-r, wy-r, wx+r, wy+r,
                          outline=CLR_WAYPOINT, width=2,
                          fill="#003300")
            self._text(c, str(i+1), wx-4, wy+5, CLR_WAYPOINT, size=9)
            prev = (wx, wy)

        # ── 드래그 중인 ROI 미리보기 ──────────────────────────
        if self._dragging and self._drag_start and self._drag_current:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            clr_roi = CLR_LEVEL_ROI if self._setup_mode == MODE_LEVEL_ROI else CLR_HP_ROI
            c.create_rectangle(min(x1,x2), min(y1,y2),
                               max(x1,x2), max(y1,y2),
                               outline=clr_roi, width=2, dash=(4,4))
            w = abs(x2-x1)
            h = abs(y2-y1)
            self._text(c, f"{w}x{h}", min(x1,x2)+4, min(y1,y2)+14, clr_roi, size=9)

        # ── 저장된 레벨/HP ROI 표시 ───────────────────────────
        self._draw_roi_box(c, self._level_roi, CLR_LEVEL_ROI, "레벨 UI")
        self._draw_roi_box(c, self._hp_roi,    CLR_HP_ROI,    "HP바")

    def _draw_registered_markers(self, c):
        """일반 모드에서도 등록된 좌표를 작게 표시."""
        # 허수아비
        if self._dummy_pos:
            dx, dy = self._dummy_pos
            r = 8
            c.create_oval(dx-r, dy-r, dx+r, dy+r,
                          outline=CLR_DUMMY, width=2)
            c.create_line(dx-r, dy, dx+r, dy, fill=CLR_DUMMY, width=1)
            c.create_line(dx, dy-r, dx, dy+r, fill=CLR_DUMMY, width=1)

        # 웨이포인트
        prev = None
        for i, (wx, wy) in enumerate(self._waypoints):
            if prev:
                c.create_line(prev[0], prev[1], wx, wy,
                              fill=CLR_WAYPOINT, width=1, dash=(4, 4))
            r = 6
            c.create_oval(wx-r, wy-r, wx+r, wy+r,
                          outline=CLR_WAYPOINT, width=1)
            self._text(c, str(i+1), wx-3, wy+4, CLR_WAYPOINT, size=7)
            prev = (wx, wy)

        # ROI 박스
        self._draw_roi_box(c, self._level_roi, CLR_LEVEL_ROI, "레벨 UI", thin=True)
        self._draw_roi_box(c, self._hp_roi,    CLR_HP_ROI,    "HP바",    thin=True)

    def _draw_roi_box(self, c, roi, color, label, thin=False):
        """ROI 딕셔너리를 박스로 그리기."""
        if not roi:
            return
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        # 오버레이 좌표로 변환 (절대좌표 → 오버레이 기준)
        ox = x - self._win_x
        oy = y - self._win_y
        lw = 1 if thin else 2
        c.create_rectangle(ox, oy, ox+w, oy+h,
                           outline=color, width=lw, dash=(6, 3))
        self._text(c, f"{label}  {w}x{h}", ox+4, oy+14, color, size=8)

    def _draw_hud(self, c, dets, target, det_fps, cap_fps, state, hp_pct, level):
        """좌상단 HUD."""
        state_color = {
            "HUNTING":      CLR_TARGET,
            "ADENA_CHECK":  CLR_ADENA_TGT,
            "DUMMY_ATTACK": CLR_DUMMY,
            "MOVING":       CLR_WAYPOINT,
        }.get(state, CLR_TEXT)

        # HP 표시
        if hp_pct is not None:
            hp_str = f"{hp_pct*100:.0f}%"
            hp_clr = "#FF4444" if hp_pct < 0.5 else "#44FF44"
        else:
            hp_str = "?"
            hp_clr = "#AAAAAA"

        level_str = str(level) if level is not None else "?"

        hud = [
            f"State    : {state}",
            f"Level    : {level_str}",
            f"HP       : {hp_str}",
            f"Monsters : {len([d for d in dets if d.class_id==0])}",
            f"Adenas   : {len([d for d in dets if d.class_id==1])}",
            f"Det FPS  : {det_fps:.1f}",
            f"Cap FPS  : {cap_fps:.1f}",
            f"Target   : {'YES' if target else 'NONE'}",
            f"ROI      : {'ON' if self._roi else 'OFF'}",
        ]
        hud_h = len(hud) * 18 + 8
        c.create_rectangle(6, 6, 210, 6 + hud_h,
                           fill="#000000", outline="", stipple="gray50")
        for i, line in enumerate(hud):
            if i == 0:
                clr = state_color
            elif i == 2:
                clr = hp_clr
            else:
                clr = CLR_TEXT
            self._text(c, line, 10, 16 + i * 18, clr, size=9)

        # 설정 모드 안내 (우하단)
        if self._setup_mode == MODE_NORMAL:
            tips = [
                "F1: 허수아비설정  F2: 웨이포인트",
                "F3: 레벨ROI  F4: HPROI  Enter: 저장",
            ]
            for i, t in enumerate(reversed(tips)):
                self._text(c, t, self._ov_w - 280,
                           self._ov_h - 12 - i * 16,
                           "#777777", size=8)

    @staticmethod
    def _text(canvas, txt, x, y, color, size=10):
        """그림자 효과 텍스트."""
        canvas.create_text(x+1, y+1, text=txt, anchor="sw",
                           fill="#000000",
                           font=("Consolas", size, "bold"))
        canvas.create_text(x, y, text=txt, anchor="sw",
                           fill=color,
                           font=("Consolas", size, "bold"))
