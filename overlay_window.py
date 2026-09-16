"""
overlay_window.py
-----------------
게임화면 위 투명 오버레이.

우상단 버튼 메뉴:
  [허수아비] → 클릭 후 게임화면 우클릭으로 허수아비 좌표 등록
  [웨이포인트] → 클릭 후 게임화면 우클릭으로 최대 10개 등록
  [레벨ROI] → 클릭 후 드래그로 레벨 UI 영역 지정
  [HP ROI]  → 클릭 후 드래그로 HP바 영역 지정
  [저장]    → config.json 저장
  [취소]    → 마지막 등록 취소
  [완료]    → 일반 모드 복귀

오버레이가 클릭 통과 ON/OFF를 자동 전환:
  일반모드    → 클릭 통과 (게임으로 전달)
  설정모드    → 클릭 수신 (오버레이가 마우스 받음)
"""

import tkinter as tk
import threading
import time
import json
import ctypes
import ctypes.wintypes
from typing import Callable

# ── Windows 클릭통과 API ─────────────────────────────────────
GWL_EXSTYLE       = -20
WS_EX_TRANSPARENT = 0x00000020

def _set_click_through(hwnd, enable: bool):
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
CLR_TEXT      = "#FFFFFF"
CLR_ROI       = "#00FFFF"
CLR_DUMMY     = "#FF4444"
CLR_WAYPOINT  = "#44FF44"
CLR_LEVEL_ROI = "#FFFF00"
CLR_HP_ROI    = "#FF8888"

TRANSPARENT_KEY = "#010101"

# ── 설정 모드 ────────────────────────────────────────────────
MODE_NORMAL    = "NORMAL"
MODE_DUMMY     = "SETUP_DUMMY"
MODE_WAYPOINT  = "SETUP_WAYPOINT"
MODE_LEVEL_ROI = "SETUP_LEVEL_ROI"
MODE_HP_ROI    = "SETUP_HP_ROI"

MAX_WAYPOINTS = 10

# ── 버튼 정의 ────────────────────────────────────────────────
# (라벨, 모드, 색상)
BUTTONS = [
    ("허수아비",   MODE_DUMMY,     CLR_DUMMY),
    ("웨이포인트", MODE_WAYPOINT,  CLR_WAYPOINT),
    ("레벨ROI",   MODE_LEVEL_ROI, CLR_LEVEL_ROI),
    ("HP ROI",    MODE_HP_ROI,    CLR_HP_ROI),
]

BTN_W  = 80   # 버튼 너비
BTN_H  = 28   # 버튼 높이
BTN_GAP = 4   # 버튼 간격
BTN_PAD = 8   # 우측/상단 여백


class OverlayWindow:

    def __init__(self,
                 mon_left: int, mon_top: int,
                 game_w: int, game_h: int,
                 lb_x: int,
                 ctrl=None,
                 roi: dict = None,
                 mon_w: int = 1920, mon_h: int = 1080,
                 config_path: str = None,
                 on_config_saved: Callable = None):

        self._mon_left        = mon_left
        self._mon_top         = mon_top
        self._game_w          = game_w
        self._game_h          = game_h
        self._lb_x            = lb_x
        self._ctrl            = ctrl
        self._mon_w           = mon_w
        self._mon_h           = mon_h
        self._config_path     = config_path
        self._on_config_saved = on_config_saved

        self._roi = roi if (roi and roi.get("enabled")) else None

        self._win_x = mon_left
        self._win_y = mon_top
        self._ov_w  = mon_w
        self._ov_h  = mon_h

        # ── 공유 데이터 ──────────────────────────────────────
        self._lock       = threading.Lock()
        self._detections = []
        self._target     = None
        self._miss_t     = 0.0
        self._det_fps    = 0.0
        self._cap_fps    = 0.0
        self._state      = "HUNTING"
        self._hp_pct     = None
        self._level      = None

        self._running = False
        self._root    = None
        self._canvas  = None
        self._thread  = None
        self._hwnd    = None

        # ── 설정 모드 상태 ───────────────────────────────────
        self._setup_mode   = MODE_NORMAL
        self._dummy_pos    = None
        self._waypoints    = []
        self._level_roi    = None
        self._hp_roi       = None

        # 드래그 상태
        self._drag_start   = None
        self._drag_current = None
        self._dragging     = False

        # 버튼 히트박스 [(x1,y1,x2,y2, mode), ...]
        self._btn_rects    = []
        # 액션 버튼 히트박스 [(x1,y1,x2,y2, action), ...]
        self._act_rects    = []

        self._last_click   = 0.0
        self._click_cd     = 0.15

    # ─────────────────────────────────────────────────────────
    #  외부 API
    # ─────────────────────────────────────────────────────────

    def update(self, detections, target=None, miss_elapsed=0.0,
               det_fps=0.0, cap_fps=0.0, state="HUNTING",
               hp_pct=None, level=None):
        with self._lock:
            self._detections = list(detections)
            self._target     = target
            self._miss_t     = miss_elapsed
            self._det_fps    = det_fps
            self._cap_fps    = cap_fps
            self._state      = state
            self._hp_pct     = hp_pct
            self._level      = level

    def load_setup(self, dummy_pos=None, waypoints=None,
                   level_roi=None, hp_roi=None):
        if dummy_pos:
            self._dummy_pos = tuple(dummy_pos)
        if waypoints:
            self._waypoints = [tuple(w) for w in waypoints]
        if level_roi:
            self._level_roi = level_roi
        if hp_roi:
            self._hp_roi = hp_roi

    def start(self):
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
            root, width=self._ov_w, height=self._ov_h,
            bg=TRANSPARENT_KEY, highlightthickness=0
        )
        self._canvas.pack()

        # HWND 획득
        root.update()
        try:
            hwnd_child = root.winfo_id()
            self._hwnd = ctypes.windll.user32.GetAncestor(hwnd_child, 2)
        except Exception:
            self._hwnd = None

        # 시작: 클릭 통과 ON
        if self._hwnd:
            _set_click_through(self._hwnd, True)

        # 마우스 바인딩 (오버레이 전체)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Button-3>",        self._on_right_click)

        self._schedule_redraw()
        root.mainloop()

    def _schedule_redraw(self):
        if not self._running:
            return
        self._redraw()
        if self._root:
            self._root.after(33, self._schedule_redraw)

    # ─────────────────────────────────────────────────────────
    #  버튼 히트박스 계산
    # ─────────────────────────────────────────────────────────

    def _calc_buttons(self):
        """우상단 버튼 히트박스 계산."""
        self._btn_rects = []
        self._act_rects = []

        # 모드 버튼 (우상단 세로 배열)
        total = len(BUTTONS)
        panel_h = total * (BTN_H + BTN_GAP) - BTN_GAP
        start_x = self._ov_w - BTN_W - BTN_PAD
        start_y = BTN_PAD

        for i, (label, mode, clr) in enumerate(BUTTONS):
            x1 = start_x
            y1 = start_y + i * (BTN_H + BTN_GAP)
            x2 = x1 + BTN_W
            y2 = y1 + BTN_H
            self._btn_rects.append((x1, y1, x2, y2, mode, label, clr))

        # 액션 버튼 (모드 버튼 아래)
        act_y = start_y + total * (BTN_H + BTN_GAP) + BTN_GAP * 3
        actions = [
            ("저장", "save",  "#44AAFF"),
            ("취소", "undo",  "#AAAAAA"),
            ("완료", "done",  "#44FF44"),
        ]
        for i, (label, action, clr) in enumerate(actions):
            x1 = start_x
            y1 = act_y + i * (BTN_H + BTN_GAP)
            x2 = x1 + BTN_W
            y2 = y1 + BTN_H
            self._act_rects.append((x1, y1, x2, y2, action, label, clr))

    def _hit_button(self, x, y):
        """클릭 좌표가 버튼 위인지 확인. (mode or action, label) 반환."""
        for (x1, y1, x2, y2, mode, label, clr) in self._btn_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return ("mode", mode, label)
        for (x1, y1, x2, y2, action, label, clr) in self._act_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return ("action", action, label)
        return None

    def _is_on_button(self, x, y):
        for (x1, y1, x2, y2, *_) in self._btn_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        for (x1, y1, x2, y2, *_) in self._act_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    # ─────────────────────────────────────────────────────────
    #  마우스 이벤트
    # ─────────────────────────────────────────────────────────

    def _on_press(self, event):
        """좌클릭 press."""
        hit = self._hit_button(event.x, event.y)
        if hit:
            kind, val, label = hit
            if kind == "mode":
                self._set_mode(val)
            elif kind == "action":
                if val == "save":
                    self._save_config()
                elif val == "undo":
                    self._undo_last()
                elif val == "done":
                    self._set_mode(MODE_NORMAL)
            return

        # 드래그 시작 (ROI 모드)
        if self._setup_mode in (MODE_LEVEL_ROI, MODE_HP_ROI):
            self._drag_start   = (event.x, event.y)
            self._drag_current = (event.x, event.y)
            self._dragging     = True
            return

        # 일반 모드: 게임 클릭 (클릭통과라 여기 안 옴 - 혹시 몰라 보존)
        if self._setup_mode == MODE_NORMAL:
            sc_x = self._win_x + event.x
            sc_y = self._win_y + event.y
            if self._ctrl and self._ctrl.is_connected:
                self._ctrl.drag_attack(sc_x, sc_y)

    def _on_drag_move(self, event):
        if self._dragging:
            self._drag_current = (event.x, event.y)

    def _on_release(self, event):
        """드래그 종료 → ROI 확정."""
        if not self._dragging or self._drag_start is None:
            return
        self._dragging = False

        x1, y1 = self._drag_start
        x2, y2 = event.x, event.y

        rx = min(x1, x2) + self._win_x
        ry = min(y1, y2) + self._win_y
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        if rw < 5 or rh < 5:
            print("[Setup] 드래그 너무 작음 → 무시")
            self._drag_start = None
            return

        roi = {"x": rx, "y": ry, "w": rw, "h": rh}
        if self._setup_mode == MODE_LEVEL_ROI:
            self._level_roi = roi
            print(f"[Setup] 레벨 UI ROI 등록: {roi}")
        elif self._setup_mode == MODE_HP_ROI:
            self._hp_roi = roi
            print(f"[Setup] HP바 ROI 등록: {roi}")

        self._drag_start = None

    def _on_right_click(self, event):
        """우클릭 → 허수아비 / 웨이포인트 등록."""
        # 버튼 위 우클릭 무시
        if self._is_on_button(event.x, event.y):
            return

        sc_x = self._win_x + event.x
        sc_y = self._win_y + event.y

        if self._setup_mode == MODE_DUMMY:
            self._dummy_pos = (sc_x, sc_y)
            print(f"[Setup] 허수아비 등록: ({sc_x},{sc_y})")

        elif self._setup_mode == MODE_WAYPOINT:
            if len(self._waypoints) < MAX_WAYPOINTS:
                self._waypoints.append((sc_x, sc_y))
                print(f"[Setup] 웨이포인트 {len(self._waypoints)} 등록: ({sc_x},{sc_y})")
            else:
                print(f"[Setup] 웨이포인트 최대 {MAX_WAYPOINTS}개 초과")

    # ─────────────────────────────────────────────────────────
    #  모드 전환
    # ─────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._setup_mode   = mode
        self._drag_start   = None
        self._drag_current = None
        self._dragging     = False

        # 클릭통과 전환
        if self._hwnd:
            click_through = (mode == MODE_NORMAL)
            _set_click_through(self._hwnd, click_through)

        if mode == MODE_NORMAL:
            print("[Setup] 완료 → 일반 모드")
        else:
            labels = {
                MODE_DUMMY:     "허수아비 모드 → 우클릭으로 좌표 등록",
                MODE_WAYPOINT:  f"웨이포인트 모드 → 우클릭으로 등록 (현재 {len(self._waypoints)}개)",
                MODE_LEVEL_ROI: "레벨ROI 모드 → 드래그로 영역 지정",
                MODE_HP_ROI:    "HP ROI 모드 → 드래그로 영역 지정",
            }
            print(f"[Setup] {labels.get(mode, mode)}")

    def _undo_last(self):
        if self._setup_mode == MODE_WAYPOINT and self._waypoints:
            removed = self._waypoints.pop()
            print(f"[Setup] 웨이포인트 삭제: {removed}  남은={len(self._waypoints)}")
        elif self._setup_mode == MODE_DUMMY:
            self._dummy_pos = None
            print("[Setup] 허수아비 좌표 초기화")
        elif self._setup_mode == MODE_LEVEL_ROI:
            self._level_roi = None
            print("[Setup] 레벨ROI 초기화")
        elif self._setup_mode == MODE_HP_ROI:
            self._hp_roi = None
            print("[Setup] HP ROI 초기화")

    def _save_config(self):
        if not self._config_path:
            print("[Setup] config_path 없음")
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = json.load(f)

            if self._dummy_pos:
                cfg.setdefault("dummy", {})["pos"] = list(self._dummy_pos)
            if self._waypoints:
                cfg.setdefault("movement", {})["waypoints"] = [list(w) for w in self._waypoints]
            if self._level_roi:
                cfg.setdefault("level_detector", {})["level_roi"] = self._level_roi
            if self._hp_roi:
                cfg.setdefault("level_detector", {})["hp_roi"] = self._hp_roi

            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            print(f"[Setup] 저장 완료")
            print(f"  허수아비: {self._dummy_pos}")
            print(f"  웨이포인트: {len(self._waypoints)}개")
            print(f"  레벨ROI: {self._level_roi}")
            print(f"  HP ROI: {self._hp_roi}")

            if self._on_config_saved:
                self._on_config_saved()
        except Exception as e:
            print(f"[Setup] 저장 실패: {e}")

    # ─────────────────────────────────────────────────────────
    #  좌표 변환
    # ─────────────────────────────────────────────────────────

    def _det_to_ov(self, dx, dy):
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

        # 버튼 히트박스 계산
        self._calc_buttons()

        # ── ROI 박스 ──────────────────────────────────────────
        if self._roi:
            rx  = self._roi["x"]
            ry  = self._roi["y"]
            rw  = self._roi["width"]
            rh  = self._roi["height"]
            rx2, ry2 = rx + rw, ry + rh
            for fill, coords in [
                ("#000000", (0, 0, self._ov_w, ry)),
                ("#000000", (0, ry2, self._ov_w, self._ov_h)),
                ("#000000", (0, ry, rx, ry2)),
                ("#000000", (rx2, ry, self._ov_w, ry2)),
            ]:
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    c.create_rectangle(*coords, fill=fill, outline="", stipple="gray25")
            c.create_rectangle(rx, ry, rx2, ry2, outline=CLR_ROI, width=2, dash=(8,4))
            self._text(c, f"ROI {rw}x{rh}", rx+4, ry+14, CLR_ROI, size=8)

        # ── 탐지 박스 ─────────────────────────────────────────
        for d in dets:
            is_tgt = (target is not None and d.x == target.x and d.y == target.y)
            if is_tgt:
                continue
            ox,  oy  = self._det_to_ov(d.x, d.y)
            ox2, oy2 = self._det_to_ov(d.x + d.w, d.y + d.h)
            ocx, ocy = self._det_to_ov(d.cx, d.cy)
            if d.class_id == 1:
                clr = CLR_ADENA_TGT if state == "ADENA_CHECK" else CLR_ADENA
                lw  = 2 if state == "ADENA_CHECK" else 1
                c.create_rectangle(ox, oy, ox2, oy2, outline=clr, width=lw)
                self._text(c, f"adena {d.confidence:.2f}", ox, oy-4, clr, size=9)
            else:
                c.create_rectangle(ox, oy, ox2, oy2, outline=CLR_BOX, width=1)
                self._text(c, f"{d.confidence:.2f}", ox, oy-4, CLR_BOX, size=9)
                c.create_oval(ocx-3, ocy-3, ocx+3, ocy+3, fill=CLR_BOX, outline="")

        # ── 타겟 박스 ─────────────────────────────────────────
        if target is not None:
            ox,  oy  = self._det_to_ov(target.x, target.y)
            ox2, oy2 = self._det_to_ov(target.x + target.w, target.y + target.h)
            ocx, ocy = self._det_to_ov(target.cx, target.cy)
            color = CLR_DEAD if miss_t > 0 else CLR_TARGET
            label = f"MISS {miss_t:.1f}s" if miss_t > 0 else f"TARGET {target.confidence:.2f}"
            thick = 1 if miss_t > 0 else 2
            c.create_rectangle(ox, oy, ox2, oy2, outline=color, width=thick)
            cs = 14
            for px, py, ddx, ddy in [(ox,ox+cs,oy,oy+cs),(ox2,ox2-cs,oy,oy+cs),
                                      (ox,ox+cs,oy2,oy2-cs),(ox2,ox2-cs,oy2,oy2-cs)]:
                c.create_line(px, oy if ddy > 0 else oy2,
                              py, oy if ddy > 0 else oy2, fill=color, width=3)
                c.create_line(px, oy if ddy > 0 else oy2,
                              px, oy+cs if ddy > 0 else oy2-cs, fill=color, width=3)
            r = 6
            c.create_oval(ocx-r, ocy-r, ocx+r, ocy+r, fill=color, outline="")
            self._text(c, label, ox, oy-10, color, size=10)

        # ── 등록된 마커 ───────────────────────────────────────
        self._draw_markers(c)

        # ── 설정모드 안내 배너 ────────────────────────────────
        if self._setup_mode != MODE_NORMAL:
            self._draw_setup_banner(c)

        # ── 우상단 버튼 ───────────────────────────────────────
        self._draw_buttons(c)

        # ── 좌상단 HUD ────────────────────────────────────────
        self._draw_hud(c, dets, target, det_fps, cap_fps, state, hp_pct, level)

    def _draw_buttons(self, c):
        """우상단 버튼 패널 그리기."""
        for (x1, y1, x2, y2, mode, label, clr) in self._btn_rects:
            active = (self._setup_mode == mode)
            bg     = clr if active else "#333333"
            border = clr
            lw     = 2 if active else 1
            c.create_rectangle(x1, y1, x2, y2,
                               fill=bg, outline=border, width=lw)
            txt_clr = "#000000" if active else clr
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            c.create_text(cx+1, cy+1, text=label, fill="#000000",
                          font=("Consolas", 8, "bold"))
            c.create_text(cx, cy, text=label, fill=txt_clr,
                          font=("Consolas", 8, "bold"))

        for (x1, y1, x2, y2, action, label, clr) in self._act_rects:
            c.create_rectangle(x1, y1, x2, y2,
                               fill="#222222", outline=clr, width=1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            c.create_text(cx+1, cy+1, text=label, fill="#000000",
                          font=("Consolas", 8, "bold"))
            c.create_text(cx, cy, text=label, fill=clr,
                          font=("Consolas", 8, "bold"))

    def _draw_setup_banner(self, c):
        """설정모드일 때 상단 안내 배너."""
        hints = {
            MODE_DUMMY:     "우클릭: 허수아비 위치 등록",
            MODE_WAYPOINT:  f"우클릭: 웨이포인트 추가 ({len(self._waypoints)}/{MAX_WAYPOINTS})  |  [취소]: 마지막 삭제",
            MODE_LEVEL_ROI: "드래그: 레벨 표시 영역 지정  |  [취소]: 초기화",
            MODE_HP_ROI:    "드래그: HP바 영역 지정  |  [취소]: 초기화",
        }
        hint = hints.get(self._setup_mode, "")
        c.create_rectangle(0, 0, self._ov_w, 36,
                           fill="#000000", outline="", stipple="gray75")
        self._text(c, f"[ 설정 모드 ]  {hint}  →  완료 후 [완료] 버튼 클릭",
                   10, 24, "#FFFF00", size=10)

        # 드래그 미리보기
        if self._dragging and self._drag_start and self._drag_current:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            clr = CLR_LEVEL_ROI if self._setup_mode == MODE_LEVEL_ROI else CLR_HP_ROI
            c.create_rectangle(min(x1,x2), min(y1,y2),
                               max(x1,x2), max(y1,y2),
                               outline=clr, width=2, dash=(4,4))
            self._text(c, f"{abs(x2-x1)}x{abs(y2-y1)}",
                       min(x1,x2)+4, min(y1,y2)+14, clr, size=9)

    def _draw_markers(self, c):
        """등록된 허수아비/웨이포인트/ROI 마커 표시."""
        # 허수아비
        if self._dummy_pos:
            dx, dy = self._dummy_pos
            # 오버레이 좌표로 변환
            ox = dx - self._win_x
            oy = dy - self._win_y
            r  = 10 if self._setup_mode == MODE_DUMMY else 7
            lw = 2  if self._setup_mode == MODE_DUMMY else 1
            c.create_oval(ox-r, oy-r, ox+r, oy+r, outline=CLR_DUMMY, width=lw)
            c.create_line(ox-r, oy, ox+r, oy, fill=CLR_DUMMY, width=lw)
            c.create_line(ox, oy-r, ox, oy+r, fill=CLR_DUMMY, width=lw)
            if self._setup_mode == MODE_DUMMY:
                self._text(c, f"허수아비({dx},{dy})", ox+r+4, oy+5, CLR_DUMMY, size=9)

        # 웨이포인트
        prev = None
        for i, (wx, wy) in enumerate(self._waypoints):
            ox = wx - self._win_x
            oy = wy - self._win_y
            if prev:
                c.create_line(prev[0], prev[1], ox, oy,
                              fill=CLR_WAYPOINT, width=1, dash=(4,3))
            r = 8 if self._setup_mode == MODE_WAYPOINT else 5
            c.create_oval(ox-r, oy-r, ox+r, oy+r,
                          outline=CLR_WAYPOINT, width=1, fill="#003300")
            self._text(c, str(i+1), ox-3, oy+4, CLR_WAYPOINT, size=7)
            prev = (ox, oy)

        # 레벨 ROI
        self._draw_roi_box(c, self._level_roi, CLR_LEVEL_ROI, "레벨UI")
        # HP ROI
        self._draw_roi_box(c, self._hp_roi, CLR_HP_ROI, "HP바")

    def _draw_roi_box(self, c, roi, color, label):
        if not roi:
            return
        ox = roi["x"] - self._win_x
        oy = roi["y"] - self._win_y
        w, h = roi["w"], roi["h"]
        c.create_rectangle(ox, oy, ox+w, oy+h,
                           outline=color, width=1, dash=(6,3))
        self._text(c, f"{label} {w}x{h}", ox+4, oy+12, color, size=8)

    def _draw_hud(self, c, dets, target, det_fps, cap_fps, state, hp_pct, level):
        """좌상단 HUD."""
        state_clr = {
            "HUNTING":      CLR_TARGET,
            "ADENA_CHECK":  CLR_ADENA_TGT,
            "DUMMY_ATTACK": CLR_DUMMY,
            "MOVING":       CLR_WAYPOINT,
        }.get(state, CLR_TEXT)

        hp_str = f"{hp_pct*100:.0f}%" if hp_pct is not None else "?"
        hp_clr = "#FF4444" if (hp_pct is not None and hp_pct < 0.5) else "#44FF44"
        lv_str = str(level) if level is not None else "?"

        hud = [
            (f"State : {state}",                           state_clr),
            (f"Level : {lv_str}",                          CLR_TEXT),
            (f"HP    : {hp_str}",                          hp_clr),
            (f"MOB   : {len([d for d in dets if d.class_id==0])}", CLR_TEXT),
            (f"ADENA : {len([d for d in dets if d.class_id==1])}", CLR_TEXT),
            (f"FPS   : {det_fps:.1f}",                     CLR_TEXT),
            (f"ROI   : {'ON' if self._roi else 'OFF'}",    CLR_TEXT),
        ]
        hud_h = len(hud) * 17 + 8
        c.create_rectangle(6, 6, 185, 6+hud_h,
                           fill="#000000", outline="", stipple="gray50")
        for i, (txt, clr) in enumerate(hud):
            self._text(c, txt, 10, 16 + i*17, clr, size=9)

    @staticmethod
    def _text(canvas, txt, x, y, color, size=10):
        canvas.create_text(x+1, y+1, text=txt, anchor="sw",
                           fill="#000000", font=("Consolas", size, "bold"))
        canvas.create_text(x, y, text=txt, anchor="sw",
                           fill=color, font=("Consolas", size, "bold"))
