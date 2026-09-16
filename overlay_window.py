"""
overlay_window.py
-----------------
게임화면 위 투명 오버레이 — 탐지 결과 표시 전용.

· 설정 기능 없음 (좌표 등록은 setup_tool.py 사용)
· 항상 클릭 통과 (WS_EX_TRANSPARENT) → 게임 조작 방해 없음
· 탐지 박스 / 타겟 박스 / HUD 정보만 표시

표시 항목:
  - 몬스터 / 아데나 탐지 박스
  - 현재 타겟 박스 (색상+십자선)
  - 좌상단 HUD (State, Level, HP%, MOB, ADENA, FPS)
  - ROI 영역 표시 (enabled 일 때)
  - 허수아비 / 웨이포인트 / 레벨ROI / HP ROI 마커 (config 로드 시)
"""

import tkinter as tk
import threading
import ctypes
import ctypes.wintypes
from typing import Optional

# ── Windows 클릭통과 API ─────────────────────────────────────────
GWL_EXSTYLE       = -20
WS_EX_TRANSPARENT = 0x00000020


def _set_click_through(hwnd, enable: bool):
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if enable:
        style |= WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


# ── 색상 ────────────────────────────────────────────────────────
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


class OverlayWindow:

    def __init__(self,
                 mon_left: int, mon_top: int,
                 game_w: int, game_h: int,
                 lb_x: int,
                 ctrl=None,
                 roi: dict = None,
                 mon_w: int = 1920, mon_h: int = 1080,
                 config_path: str = None,
                 on_config_saved=None):   # 호환성 유지용 (사용 안 함)

        self._mon_left = mon_left
        self._mon_top  = mon_top
        self._game_w   = game_w
        self._game_h   = game_h
        self._lb_x     = lb_x
        self._ctrl     = ctrl
        self._mon_w    = mon_w
        self._mon_h    = mon_h

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

        # ── 마커 표시용 (config 로드 후 set) ─────────────────
        self._dummy_pos  = None   # (abs_x, abs_y)
        self._waypoints  = []     # [(abs_x, abs_y), ...]
        self._level_roi  = None   # {"x","y","w","h"}
        self._hp_roi     = None   # {"x","y","w","h"}

    # ─────────────────────────────────────────────────────────────
    #  외부 API
    # ─────────────────────────────────────────────────────────────

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
        """config에서 로드한 마커 정보를 오버레이에 전달."""
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

    # ─────────────────────────────────────────────────────────────
    #  tkinter 루프
    # ─────────────────────────────────────────────────────────────

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

        # HWND 획득 → 클릭 통과 ON (항상)
        root.update()
        try:
            hwnd_child = root.winfo_id()
            self._hwnd = ctypes.windll.user32.GetAncestor(hwnd_child, 2)
        except Exception:
            self._hwnd = None

        if self._hwnd:
            _set_click_through(self._hwnd, True)

        self._schedule_redraw()
        root.mainloop()

    def _schedule_redraw(self):
        if not self._running:
            return
        self._redraw()
        if self._root:
            self._root.after(33, self._schedule_redraw)

    # ─────────────────────────────────────────────────────────────
    #  좌표 변환
    # ─────────────────────────────────────────────────────────────

    def _det_to_ov(self, dx, dy):
        if self._roi:
            return dx + self._roi["x"], dy + self._roi["y"]
        return dx - self._lb_x, dy

    # ─────────────────────────────────────────────────────────────
    #  그리기
    # ─────────────────────────────────────────────────────────────

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
            rx2, ry2 = rx + rw, ry + rh
            for fill, coords in [
                ("#000000", (0, 0, self._ov_w, ry)),
                ("#000000", (0, ry2, self._ov_w, self._ov_h)),
                ("#000000", (0, ry, rx, ry2)),
                ("#000000", (rx2, ry, self._ov_w, ry2)),
            ]:
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    c.create_rectangle(*coords, fill=fill, outline="",
                                       stipple="gray25")
            c.create_rectangle(rx, ry, rx2, ry2,
                               outline=CLR_ROI, width=2, dash=(8, 4))
            self._text(c, f"ROI {rw}x{rh}", rx+4, ry+14, CLR_ROI, size=8)

        # ── 탐지 박스 ─────────────────────────────────────────
        for d in dets:
            is_tgt = (target is not None
                      and d.x == target.x and d.y == target.y)
            if is_tgt:
                continue
            ox,  oy  = self._det_to_ov(d.x, d.y)
            ox2, oy2 = self._det_to_ov(d.x + d.w, d.y + d.h)
            ocx, ocy = self._det_to_ov(d.cx, d.cy)
            if d.class_id == 1:
                clr = CLR_ADENA_TGT if state == "LOOTING" else CLR_ADENA
                lw  = 2 if state == "LOOTING" else 1
                c.create_rectangle(ox, oy, ox2, oy2, outline=clr, width=lw)
                self._text(c, f"adena {d.confidence:.2f}",
                           ox, oy-4, clr, size=9)
            else:
                c.create_rectangle(ox, oy, ox2, oy2,
                                   outline=CLR_BOX, width=1)
                self._text(c, f"{d.confidence:.2f}",
                           ox, oy-4, CLR_BOX, size=9)
                c.create_oval(ocx-3, ocy-3, ocx+3, ocy+3,
                              fill=CLR_BOX, outline="")

        # ── 타겟 박스 ─────────────────────────────────────────
        if target is not None:
            ox,  oy  = self._det_to_ov(target.x, target.y)
            ox2, oy2 = self._det_to_ov(target.x + target.w, target.y + target.h)
            ocx, ocy = self._det_to_ov(target.cx, target.cy)
            color = CLR_DEAD if miss_t > 0 else CLR_TARGET
            label = (f"MISS {miss_t:.1f}s" if miss_t > 0
                     else f"TARGET {target.confidence:.2f}")
            thick = 1 if miss_t > 0 else 2
            c.create_rectangle(ox, oy, ox2, oy2, outline=color, width=thick)
            cs = 14
            corners = [
                (ox,  ox+cs,  oy,  oy+cs),
                (ox2, ox2-cs, oy,  oy+cs),
                (ox,  ox+cs,  oy2, oy2-cs),
                (ox2, ox2-cs, oy2, oy2-cs),
            ]
            for bx, ex, by, ey in corners:
                top = (by < ey)
                base_y = oy if top else oy2
                c.create_line(bx, base_y, ex, base_y,
                              fill=color, width=3)
                c.create_line(bx, base_y, bx,
                              base_y + cs if top else base_y - cs,
                              fill=color, width=3)
            r = 6
            c.create_oval(ocx-r, ocy-r, ocx+r, ocy+r,
                          fill=color, outline="")
            self._text(c, label, ox, oy-10, color, size=10)

        # ── 마커 (config 로드된 좌표) ─────────────────────────
        self._draw_markers(c)

        # ── HUD ───────────────────────────────────────────────
        self._draw_hud(c, dets, target, det_fps, cap_fps,
                       state, hp_pct, level)

    # ─────────────────────────────────────────────────────────────
    #  마커 렌더링
    # ─────────────────────────────────────────────────────────────

    def _draw_markers(self, c):
        """허수아비 / 웨이포인트 / 레벨ROI / HP ROI 마커."""
        # 허수아비
        if self._dummy_pos:
            dx, dy = self._dummy_pos
            ox = dx - self._win_x
            oy = dy - self._win_y
            r  = 7
            c.create_oval(ox-r, oy-r, ox+r, oy+r,
                          outline=CLR_DUMMY, width=1)
            c.create_line(ox-r, oy, ox+r, oy, fill=CLR_DUMMY, width=1)
            c.create_line(ox, oy-r, ox, oy+r, fill=CLR_DUMMY, width=1)

        # 웨이포인트
        prev = None
        for i, (wx, wy) in enumerate(self._waypoints):
            ox = wx - self._win_x
            oy = wy - self._win_y
            if prev:
                c.create_line(prev[0], prev[1], ox, oy,
                              fill=CLR_WAYPOINT, width=1, dash=(4, 3))
            r = 5
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
                           outline=color, width=1, dash=(6, 3))
        self._text(c, f"{label} {w}x{h}", ox+4, oy+12, color, size=8)

    # ─────────────────────────────────────────────────────────────
    #  HUD
    # ─────────────────────────────────────────────────────────────

    def _draw_hud(self, c, dets, target, det_fps, cap_fps,
                  state, hp_pct, level):
        state_clr = {
            "HUNTING":           CLR_TARGET,
            "LOOTING":           CLR_ADENA_TGT,
            "ATTACKING_DUMMY":   CLR_DUMMY,
            "MOVE_TO_HUNT_ZONE": CLR_WAYPOINT,
            "IDLE":              CLR_TEXT,
            "DONE":              CLR_TEXT,
        }.get(state, CLR_TEXT)

        hp_str = f"{hp_pct:.1f}%" if hp_pct is not None else "?"
        hp_clr = ("#FF4444" if (hp_pct is not None and hp_pct < 50.0)  # 0~100% 범위
                  else "#44FF44")
        lv_str = str(level) if level is not None else "?"

        hud = [
            (f"State : {state}",                                state_clr),
            (f"Level : {lv_str}",                               CLR_TEXT),
            (f"HP    : {hp_str}",                               hp_clr),
            (f"MOB   : {len([d for d in dets if d.class_id==0])}", CLR_TEXT),
            (f"ADENA : {len([d for d in dets if d.class_id==1])}", CLR_TEXT),
            (f"FPS   : {det_fps:.1f}",                          CLR_TEXT),
            (f"ROI   : {'ON' if self._roi else 'OFF'}",         CLR_TEXT),
        ]
        hud_h = len(hud) * 17 + 8
        c.create_rectangle(6, 6, 185, 6+hud_h,
                           fill="#000000", outline="", stipple="gray50")
        for i, (txt, clr) in enumerate(hud):
            self._text(c, txt, 10, 16 + i*17, clr, size=9)

    # ─────────────────────────────────────────────────────────────
    #  텍스트 헬퍼
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _text(canvas, txt, x, y, color, size=10):
        canvas.create_text(x+1, y+1, text=txt, anchor="sw",
                           fill="#000000",
                           font=("Consolas", size, "bold"))
        canvas.create_text(x, y, text=txt, anchor="sw",
                           fill=color,
                           font=("Consolas", size, "bold"))
