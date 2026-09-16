"""
setup_tool.py
-------------
별도 설정 창 — 게임화면 캡처 후 이미지 위 클릭/드래그로 좌표 등록.

사용법:
  python setup_tool.py

기능:
  [스크린샷]      → 현재 화면 캡처 (mss)
  [허수아비 설정] → 이미지 클릭 → 절대좌표 등록
  [웨이포인트]    → 이미지 순서대로 클릭 → 최대 10개
  [레벨 ROI]      → 이미지 드래그 → 레벨 UI 영역
  [HP ROI]        → 이미지 드래그 → HP바 영역
  [저장]          → config.json 저장 후 완료

투명 오버레이가 아닌 일반 창이므로 클릭 문제 없음.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import threading
import time
import sys

try:
    import mss
    import mss.tools
except ImportError:
    print("[SetupTool] mss 없음 → pip install mss")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except ImportError:
    print("[SetupTool] Pillow 없음 → pip install Pillow")
    sys.exit(1)

# ── 상수 ────────────────────────────────────────────────────────
MAX_WAYPOINTS = 10
CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "config.json")

CLR_DUMMY    = "#FF4444"
CLR_WP       = "#44FF44"
CLR_LV_ROI   = "#FFFF00"
CLR_HP_ROI   = "#FF8888"
CLR_INFO     = "#00CCFF"

# ── 모드 상수 ────────────────────────────────────────────────────
MODE_NONE     = "NONE"
MODE_DUMMY    = "DUMMY"
MODE_WAYPOINT = "WAYPOINT"
MODE_LV_ROI   = "LV_ROI"
MODE_HP_ROI   = "HP_ROI"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class SetupTool:
    def __init__(self):
        # ── 등록된 좌표 ─────────────────────────────────────
        self._dummy_pos  = None   # (abs_x, abs_y)
        self._waypoints  = []     # [(abs_x, abs_y), ...]
        self._level_roi  = None   # {"x","y","w","h"}
        self._hp_roi     = None   # {"x","y","w","h"}

        # ── 캡처 이미지 상태 ────────────────────────────────
        self._screenshot     = None   # PIL.Image (원본 전체화면)
        self._cap_mon        = None   # mss 모니터 dict
        self._scale          = 1.0    # 표시 스케일
        self._img_offset_x   = 0      # 캔버스 내 이미지 시작 픽셀
        self._img_offset_y   = 0
        self._display_img    = None   # PIL.Image (리사이즈 후)
        self._tk_img         = None   # ImageTk.PhotoImage

        # ── 현재 모드 ────────────────────────────────────────
        self._mode       = MODE_NONE

        # 드래그 상태
        self._drag_start  = None   # (canvas_x, canvas_y)
        self._drag_rect   = None   # 캔버스 드래그 사각형 id

        # ── 기존 config 로드 ────────────────────────────────
        try:
            cfg = _load_config()
            dp  = cfg.get("dummy", {}).get("pos")
            if dp:
                self._dummy_pos = tuple(dp)
            wps = cfg.get("movement", {}).get("waypoints", [])
            self._waypoints = [tuple(w) for w in wps]
            lr  = cfg.get("level_detector", {}).get("level_roi")
            hr  = cfg.get("level_detector", {}).get("hp_roi")
            if lr:
                self._level_roi = lr
            if hr:
                self._hp_roi = hr
            print(f"[SetupTool] config 로드 완료")
            print(f"  허수아비: {self._dummy_pos}")
            print(f"  웨이포인트: {len(self._waypoints)}개")
            print(f"  레벨ROI: {self._level_roi}")
            print(f"  HP ROI : {self._hp_roi}")
        except Exception as e:
            print(f"[SetupTool] config 로드 실패: {e}")

        # ── UI 생성 ─────────────────────────────────────────
        self._root = tk.Tk()
        self._root.title("리니지봇 설정 툴")
        self._root.configure(bg="#1E1E1E")
        self._root.resizable(True, True)

        self._build_ui()

        # 모니터 목록
        with mss.mss() as sct:
            self._monitors = sct.monitors[1:]   # [0] = 전체합성, 제외

        # 기존 좌표 있으면 상태바 표시
        self._update_status()

    # ─────────────────────────────────────────────────────────────
    #  UI 빌드
    # ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self._root

        # ── 좌측 컨트롤 패널 ─────────────────────────────────
        left = tk.Frame(root, bg="#1E1E1E", width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        left.pack_propagate(False)

        # 제목
        tk.Label(left, text="🎮 설정 툴",
                 bg="#1E1E1E", fg="#FFFFFF",
                 font=("Consolas", 13, "bold")).pack(pady=(6, 2))
        tk.Label(left, text="config.json 편집",
                 bg="#1E1E1E", fg="#888888",
                 font=("Consolas", 8)).pack(pady=(0, 10))

        sep = lambda: tk.Frame(left, bg="#333333", height=1).pack(
            fill=tk.X, pady=4)

        # ── 캡처 버튼 ────────────────────────────────────────
        sep()
        tk.Label(left, text="① 화면 캡처",
                 bg="#1E1E1E", fg="#AAAAAA",
                 font=("Consolas", 8)).pack()

        # 모니터 선택
        mon_row = tk.Frame(left, bg="#1E1E1E")
        mon_row.pack(fill=tk.X, pady=2)
        tk.Label(mon_row, text="모니터:",
                 bg="#1E1E1E", fg="#CCCCCC",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._mon_var = tk.StringVar(value="1")
        tk.Spinbox(mon_row, from_=1, to=4, width=3,
                   textvariable=self._mon_var,
                   font=("Consolas", 9),
                   bg="#333333", fg="#FFFFFF",
                   insertbackground="#FFFFFF").pack(side=tk.LEFT, padx=4)

        self._btn_capture = self._btn(left, "📷  스크린샷", "#2255AA",
                                      self._do_capture)
        self._btn_capture.pack(fill=tk.X, pady=2)

        # ── 등록 버튼들 ──────────────────────────────────────
        sep()
        tk.Label(left, text="② 좌표 등록",
                 bg="#1E1E1E", fg="#AAAAAA",
                 font=("Consolas", 8)).pack()

        self._btn_dummy = self._btn(left, "🗡  허수아비 설정", CLR_DUMMY,
                                    lambda: self._set_mode(MODE_DUMMY))
        self._btn_dummy.pack(fill=tk.X, pady=2)

        self._btn_wp = self._btn(left, "📍  웨이포인트", CLR_WP,
                                 lambda: self._set_mode(MODE_WAYPOINT))
        self._btn_wp.pack(fill=tk.X, pady=2)

        self._btn_lroi = self._btn(left, "🔵  레벨 ROI", CLR_LV_ROI,
                                   lambda: self._set_mode(MODE_LV_ROI))
        self._btn_lroi.pack(fill=tk.X, pady=2)

        self._btn_hroi = self._btn(left, "❤  HP ROI", CLR_HP_ROI,
                                   lambda: self._set_mode(MODE_HP_ROI))
        self._btn_hroi.pack(fill=tk.X, pady=2)

        # ── 취소/초기화 버튼 ─────────────────────────────────
        sep()
        tk.Label(left, text="③ 수정",
                 bg="#1E1E1E", fg="#AAAAAA",
                 font=("Consolas", 8)).pack()

        self._btn_undo = self._btn(left, "↩  마지막 취소", "#666666",
                                   self._do_undo)
        self._btn_undo.pack(fill=tk.X, pady=2)

        self._btn_clear = self._btn(left, "🗑  현재모드 초기화", "#884444",
                                    self._do_clear)
        self._btn_clear.pack(fill=tk.X, pady=2)

        # ── 저장 버튼 ─────────────────────────────────────────
        sep()
        self._btn_save = self._btn(left, "💾  저장 (config.json)", "#22AA44",
                                   self._do_save, font_size=10)
        self._btn_save.pack(fill=tk.X, pady=4)

        # ── 상태 정보 ─────────────────────────────────────────
        sep()
        tk.Label(left, text="등록 현황:",
                 bg="#1E1E1E", fg="#AAAAAA",
                 font=("Consolas", 8)).pack(anchor="w")
        self._lbl_status = tk.Label(left, text="",
                                    bg="#1E1E1E", fg="#CCCCCC",
                                    font=("Consolas", 8),
                                    justify=tk.LEFT, anchor="w",
                                    wraplength=185)
        self._lbl_status.pack(fill=tk.X, pady=2)

        # ── 모드 안내 ─────────────────────────────────────────
        sep()
        self._lbl_mode = tk.Label(left, text="모드: 없음",
                                   bg="#1E1E1E", fg=CLR_INFO,
                                   font=("Consolas", 9, "bold"),
                                   wraplength=185, justify=tk.LEFT)
        self._lbl_mode.pack(fill=tk.X, pady=2)

        self._lbl_hint = tk.Label(left, text="스크린샷을 먼저 캡처하세요.",
                                   bg="#1E1E1E", fg="#888888",
                                   font=("Consolas", 8),
                                   wraplength=185, justify=tk.LEFT)
        self._lbl_hint.pack(fill=tk.X, pady=2)

        # ── 우측 이미지 뷰어 ─────────────────────────────────
        right = tk.Frame(root, bg="#111111")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 캔버스 + 스크롤바
        self._canvas = tk.Canvas(right, bg="#111111",
                                 cursor="crosshair",
                                 highlightthickness=1,
                                 highlightbackground="#333333")
        vbar = ttk.Scrollbar(right, orient=tk.VERTICAL,
                             command=self._canvas.yview)
        hbar = ttk.Scrollbar(right, orient=tk.HORIZONTAL,
                             command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=vbar.set,
                                xscrollcommand=hbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 마우스 이벤트
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>",          self._on_motion)
        # 마우스 휠 줌
        self._canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self._canvas.bind("<Configure>",          self._on_resize)

        # 좌표 표시 바
        self._lbl_coord = tk.Label(right,
                                   text="마우스를 이미지 위로 이동하세요",
                                   bg="#111111", fg="#666666",
                                   font=("Consolas", 8))
        self._lbl_coord.pack(side=tk.BOTTOM, fill=tk.X)

        # 창 최소 크기
        root.minsize(900, 550)
        root.geometry("1280x720")

    @staticmethod
    def _btn(parent, text, color, cmd, font_size=9):
        return tk.Button(parent, text=text,
                         bg="#2A2A2A", fg=color,
                         activebackground="#444444", activeforeground=color,
                         font=("Consolas", font_size, "bold"),
                         relief=tk.FLAT, bd=0, padx=6, pady=5,
                         cursor="hand2",
                         command=cmd)

    # ─────────────────────────────────────────────────────────────
    #  캡처
    # ─────────────────────────────────────────────────────────────

    def _do_capture(self):
        """mss로 선택 모니터 캡처 → 이미지 뷰어에 표시."""
        try:
            mon_idx = int(self._mon_var.get())
        except ValueError:
            mon_idx = 1

        def _capture():
            try:
                with mss.mss() as sct:
                    monitors = sct.monitors
                    if mon_idx >= len(monitors):
                        # fallback to first physical monitor
                        mon = monitors[1]
                    else:
                        mon = monitors[mon_idx]
                    self._cap_mon = mon
                    shot = sct.grab(mon)
                    img  = Image.frombytes("RGB", shot.size, shot.bgra,
                                          "raw", "BGRX")
                self._screenshot = img
                self._root.after(0, self._show_screenshot)
                print(f"[Capture] 모니터 {mon_idx}: "
                      f"{mon['width']}x{mon['height']} "
                      f"left={mon['left']} top={mon['top']}")
            except Exception as e:
                print(f"[Capture] 실패: {e}")
                self._root.after(0, lambda: messagebox.showerror(
                    "캡처 실패", str(e)))

        threading.Thread(target=_capture, daemon=True).start()
        self._lbl_hint["text"] = "캡처 중..."

    def _show_screenshot(self):
        """캡처 이미지를 1:1 원본 크기로 표시 (스크롤로 이동)."""
        if self._screenshot is None:
            return

        img = self._screenshot
        iw, ih = img.size

        # 항상 1:1 원본 크기
        self._scale = 1.0
        self._display_img = img
        self._tk_img = ImageTk.PhotoImage(self._display_img)

        # 이미지는 (0,0)에 배치, 스크롤로 이동
        self._img_offset_x = 0
        self._img_offset_y = 0

        c = self._canvas
        c.delete("all")
        c.configure(scrollregion=(0, 0, iw, ih))
        c.create_image(0, 0, anchor=tk.NW, image=self._tk_img, tags="bg")

        self._redraw_markers()
        self._lbl_hint["text"] = f"원본크기 {iw}x{ih} — 스크롤로 이동, Ctrl+휠로 줌."

    def _on_resize(self, event):
        if self._screenshot:
            self._show_screenshot()

    # ─────────────────────────────────────────────────────────────
    #  줌 (Ctrl + 마우스휠)
    # ─────────────────────────────────────────────────────────────

    def _on_zoom(self, event):
        if self._screenshot is None:
            return
        factor = 1.1 if event.delta > 0 else (1 / 1.1)
        new_scale = max(0.1, min(3.0, self._scale * factor))
        if abs(new_scale - self._scale) < 0.001:
            return
        self._scale = new_scale

        img = self._screenshot
        iw, ih = img.size
        nw = max(1, int(iw * self._scale))
        nh = max(1, int(ih * self._scale))
        self._display_img = img.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(self._display_img)

        c = self._canvas
        c.update_idletasks()
        cw = c.winfo_width()
        ch = c.winfo_height()
        self._img_offset_x = max((cw - nw) // 2, 0)
        self._img_offset_y = max((ch - nh) // 2, 0)

        c.delete("all")
        c.configure(scrollregion=(0, 0,
                                   max(cw, nw + self._img_offset_x * 2),
                                   max(ch, nh + self._img_offset_y * 2)))
        c.create_image(self._img_offset_x, self._img_offset_y,
                       anchor=tk.NW, image=self._tk_img, tags="bg")
        self._redraw_markers()

    # ─────────────────────────────────────────────────────────────
    #  좌표 변환 헬퍼
    # ─────────────────────────────────────────────────────────────

    def _canvas_to_abs(self, cx, cy):
        """캔버스 픽셀 → 절대 화면 좌표."""
        if self._cap_mon is None or self._display_img is None:
            return None, None
        # 이미지 내 픽셀
        ix = cx - self._img_offset_x
        iy = cy - self._img_offset_y
        if ix < 0 or iy < 0:
            return None, None
        dw, dh = self._display_img.size
        if ix > dw or iy > dh:
            return None, None
        # 원본 이미지 좌표
        ox = ix / self._scale
        oy = iy / self._scale
        # 절대 화면 좌표
        ax = int(self._cap_mon["left"] + ox)
        ay = int(self._cap_mon["top"]  + oy)
        return ax, ay

    def _abs_to_canvas(self, ax, ay):
        """절대 화면 좌표 → 캔버스 픽셀."""
        if self._cap_mon is None:
            return None, None
        ox = ax - self._cap_mon["left"]
        oy = ay - self._cap_mon["top"]
        cx = int(ox * self._scale) + self._img_offset_x
        cy = int(oy * self._scale) + self._img_offset_y
        return cx, cy

    def _roi_to_canvas(self, roi):
        """roi {x,y,w,h} → 캔버스 (x1,y1,x2,y2)."""
        if self._cap_mon is None:
            return None
        x1, y1 = self._abs_to_canvas(roi["x"], roi["y"])
        x2 = x1 + int(roi["w"] * self._scale)
        y2 = y1 + int(roi["h"] * self._scale)
        if x1 is None:
            return None
        return x1, y1, x2, y2

    # ─────────────────────────────────────────────────────────────
    #  마우스 이벤트
    # ─────────────────────────────────────────────────────────────

    def _on_motion(self, event):
        """마우스 이동 → 좌표 표시."""
        ax, ay = self._canvas_to_abs(event.x, event.y)
        if ax is not None:
            self._lbl_coord["text"] = f"화면 좌표: ({ax}, {ay})   캔버스: ({event.x}, {event.y})"
        else:
            self._lbl_coord["text"] = f"캔버스: ({event.x}, {event.y})"

    def _on_press(self, event):
        """좌클릭 press."""
        if self._screenshot is None:
            self._lbl_hint["text"] = "⚠ 먼저 [스크린샷] 버튼으로 화면을 캡처하세요!"
            return

        ax, ay = self._canvas_to_abs(event.x, event.y)
        if ax is None:
            return  # 이미지 바깥

        if self._mode == MODE_DUMMY:
            self._dummy_pos = (ax, ay)
            print(f"[Setup] 허수아비 등록: ({ax},{ay})")
            self._redraw_markers()
            self._update_status()

        elif self._mode == MODE_WAYPOINT:
            if len(self._waypoints) < MAX_WAYPOINTS:
                self._waypoints.append((ax, ay))
                print(f"[Setup] 웨이포인트 {len(self._waypoints)} 등록: ({ax},{ay})")
                self._redraw_markers()
                self._update_status()
                self._lbl_hint["text"] = (
                    f"웨이포인트 {len(self._waypoints)}/{MAX_WAYPOINTS} 등록됨. "
                    "계속 클릭하거나 [저장]을 누르세요."
                )
            else:
                self._lbl_hint["text"] = f"최대 {MAX_WAYPOINTS}개 초과!"

        elif self._mode in (MODE_LV_ROI, MODE_HP_ROI):
            # 드래그 시작
            self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        """드래그 중 → 사각형 미리보기."""
        if self._mode not in (MODE_LV_ROI, MODE_HP_ROI):
            return
        if self._drag_start is None:
            return

        # 기존 드래그 rect 삭제
        if self._drag_rect is not None:
            self._canvas.delete(self._drag_rect)

        x1, y1 = self._drag_start
        x2, y2 = event.x, event.y
        clr = CLR_LV_ROI if self._mode == MODE_LV_ROI else CLR_HP_ROI
        self._drag_rect = self._canvas.create_rectangle(
            min(x1, x2), min(y1, y2),
            max(x1, x2), max(y1, y2),
            outline=clr, width=2, dash=(5, 3)
        )
        # 크기 표시
        ax1, ay1 = self._canvas_to_abs(min(x1,x2), min(y1,y2))
        ax2, ay2 = self._canvas_to_abs(max(x1,x2), max(y1,y2))
        if ax1 and ax2:
            w = ax2 - ax1
            h = ay2 - ay1
            self._lbl_coord["text"] = f"드래그 중: ({ax1},{ay1}) ~ ({ax2},{ay2})  크기={w}x{h}"

    def _on_release(self, event):
        """드래그 종료 → ROI 확정."""
        if self._mode not in (MODE_LV_ROI, MODE_HP_ROI):
            self._drag_start = None
            return
        if self._drag_start is None:
            return

        # 드래그 미리보기 rect 삭제
        if self._drag_rect is not None:
            self._canvas.delete(self._drag_rect)
            self._drag_rect = None

        x1c, y1c = self._drag_start
        x2c, y2c = event.x, event.y
        self._drag_start = None

        # 최소 크기 체크
        if abs(x2c - x1c) < 5 or abs(y2c - y1c) < 5:
            self._lbl_hint["text"] = "⚠ 드래그가 너무 작습니다. 다시 시도하세요."
            return

        # 절대 좌표
        ax1, ay1 = self._canvas_to_abs(min(x1c, x2c), min(y1c, y2c))
        ax2, ay2 = self._canvas_to_abs(max(x1c, x2c), max(y1c, y2c))
        if ax1 is None or ax2 is None:
            return

        roi = {"x": ax1, "y": ay1, "w": ax2 - ax1, "h": ay2 - ay1}

        if self._mode == MODE_LV_ROI:
            self._level_roi = roi
            print(f"[Setup] 레벨 ROI 등록: {roi}")
        elif self._mode == MODE_HP_ROI:
            self._hp_roi = roi
            print(f"[Setup] HP ROI 등록: {roi}")

        self._redraw_markers()
        self._update_status()

    # ─────────────────────────────────────────────────────────────
    #  모드 전환
    # ─────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        if self._screenshot is None and mode != MODE_NONE:
            messagebox.showinfo("안내", "먼저 [스크린샷] 버튼으로 화면을 캡처하세요!")
            return

        self._mode = mode
        self._drag_start = None
        self._drag_rect  = None

        hints = {
            MODE_NONE:     ("모드: 없음",       "버튼을 눌러 등록 모드를 선택하세요."),
            MODE_DUMMY:    ("모드: 허수아비",    "이미지를 클릭해 허수아비 위치를 등록하세요."),
            MODE_WAYPOINT: ("모드: 웨이포인트",  f"이미지를 순서대로 클릭 ({len(self._waypoints)}/{MAX_WAYPOINTS}개)."),
            MODE_LV_ROI:   ("모드: 레벨 ROI",   "이미지 위에서 드래그해 레벨 UI 영역을 지정하세요."),
            MODE_HP_ROI:   ("모드: HP ROI",     "이미지 위에서 드래그해 HP바 영역을 지정하세요."),
        }
        m_txt, h_txt = hints.get(mode, ("", ""))
        self._lbl_mode["text"] = m_txt
        self._lbl_hint["text"] = h_txt

        # 버튼 하이라이트
        color_map = {
            MODE_DUMMY:    (self._btn_dummy,  CLR_DUMMY),
            MODE_WAYPOINT: (self._btn_wp,     CLR_WP),
            MODE_LV_ROI:   (self._btn_lroi,   CLR_LV_ROI),
            MODE_HP_ROI:   (self._btn_hroi,   CLR_HP_ROI),
        }
        for m, (btn, clr) in color_map.items():
            if m == mode:
                btn.configure(bg=clr, fg="#000000")
            else:
                btn.configure(bg="#2A2A2A", fg=clr)

    # ─────────────────────────────────────────────────────────────
    #  취소 / 초기화
    # ─────────────────────────────────────────────────────────────

    def _do_undo(self):
        if self._mode == MODE_WAYPOINT and self._waypoints:
            removed = self._waypoints.pop()
            print(f"[Setup] 웨이포인트 삭제: {removed}  남은={len(self._waypoints)}")
            self._lbl_hint["text"] = f"웨이포인트 삭제됨. 남은={len(self._waypoints)}개"
        elif self._mode == MODE_DUMMY:
            self._dummy_pos = None
            print("[Setup] 허수아비 초기화")
            self._lbl_hint["text"] = "허수아비 좌표 삭제됨"
        elif self._mode == MODE_LV_ROI:
            self._level_roi = None
            print("[Setup] 레벨 ROI 초기화")
            self._lbl_hint["text"] = "레벨 ROI 삭제됨"
        elif self._mode == MODE_HP_ROI:
            self._hp_roi = None
            print("[Setup] HP ROI 초기화")
            self._lbl_hint["text"] = "HP ROI 삭제됨"
        else:
            self._lbl_hint["text"] = "취소할 항목이 없습니다."
        self._redraw_markers()
        self._update_status()

    def _do_clear(self):
        if self._mode == MODE_DUMMY:
            self._dummy_pos = None
        elif self._mode == MODE_WAYPOINT:
            self._waypoints.clear()
        elif self._mode == MODE_LV_ROI:
            self._level_roi = None
        elif self._mode == MODE_HP_ROI:
            self._hp_roi = None
        else:
            self._lbl_hint["text"] = "초기화할 항목이 없습니다."
            return
        self._lbl_hint["text"] = "초기화 완료"
        self._redraw_markers()
        self._update_status()

    # ─────────────────────────────────────────────────────────────
    #  저장
    # ─────────────────────────────────────────────────────────────

    def _do_save(self):
        try:
            cfg = _load_config()

            # 허수아비
            if self._dummy_pos:
                cfg.setdefault("dummy", {})["pos"] = list(self._dummy_pos)

            # 웨이포인트
            if self._waypoints:
                cfg.setdefault("movement", {})["waypoints"] = [
                    list(w) for w in self._waypoints
                ]

            # 레벨 ROI
            if self._level_roi:
                cfg.setdefault("level_detector", {})["level_roi"] = self._level_roi

            # HP ROI
            if self._hp_roi:
                cfg.setdefault("level_detector", {})["hp_roi"] = self._hp_roi

            _save_config(cfg)

            summary = (
                f"저장 완료!\n\n"
                f"허수아비: {self._dummy_pos}\n"
                f"웨이포인트: {len(self._waypoints)}개\n"
                f"레벨 ROI: {self._level_roi}\n"
                f"HP ROI: {self._hp_roi}"
            )
            print("[Setup] config.json 저장 완료")
            print(f"  허수아비: {self._dummy_pos}")
            print(f"  웨이포인트: {len(self._waypoints)}개")
            print(f"  레벨 ROI: {self._level_roi}")
            print(f"  HP ROI: {self._hp_roi}")
            messagebox.showinfo("저장 완료", summary)
            self._lbl_hint["text"] = "✅ 저장 완료! 이제 main.py를 실행하세요."

        except Exception as e:
            print(f"[Setup] 저장 실패: {e}")
            messagebox.showerror("저장 실패", str(e))

    # ─────────────────────────────────────────────────────────────
    #  상태 표시
    # ─────────────────────────────────────────────────────────────

    def _update_status(self):
        dummy_str = f"({self._dummy_pos[0]},{self._dummy_pos[1]})" \
                    if self._dummy_pos else "미설정"
        wp_str    = f"{len(self._waypoints)}개" \
                    if self._waypoints else "없음"
        lr_str    = f"{self._level_roi['w']}x{self._level_roi['h']}" \
                    if self._level_roi else "미설정"
        hr_str    = f"{self._hp_roi['w']}x{self._hp_roi['h']}" \
                    if self._hp_roi else "미설정"
        self._lbl_status["text"] = (
            f"허수아비: {dummy_str}\n"
            f"웨이포인트: {wp_str}\n"
            f"레벨ROI: {lr_str}\n"
            f"HP ROI: {hr_str}"
        )

        # 웨이포인트 모드 힌트 갱신
        if self._mode == MODE_WAYPOINT:
            self._lbl_hint["text"] = (
                f"이미지를 순서대로 클릭 ({len(self._waypoints)}/{MAX_WAYPOINTS}개)."
            )

    # ─────────────────────────────────────────────────────────────
    #  마커 렌더링
    # ─────────────────────────────────────────────────────────────

    def _redraw_markers(self):
        """캔버스에 등록된 마커를 다시 그림."""
        c = self._canvas
        c.delete("marker")   # 마커 태그만 삭제

        r = 8  # 기본 반경

        # ── 허수아비 ──────────────────────────────────────────
        if self._dummy_pos:
            cx, cy = self._abs_to_canvas(*self._dummy_pos)
            if cx is not None:
                c.create_oval(cx-r, cy-r, cx+r, cy+r,
                              outline=CLR_DUMMY, width=2,
                              tags="marker")
                c.create_line(cx-r, cy, cx+r, cy,
                              fill=CLR_DUMMY, width=2, tags="marker")
                c.create_line(cx, cy-r, cx, cy+r,
                              fill=CLR_DUMMY, width=2, tags="marker")
                c.create_text(cx+r+4, cy,
                              text=f"허수아비 ({self._dummy_pos[0]},{self._dummy_pos[1]})",
                              anchor=tk.W, fill=CLR_DUMMY,
                              font=("Consolas", 8, "bold"), tags="marker")

        # ── 웨이포인트 ────────────────────────────────────────
        prev_cc = None
        for i, (ax, ay) in enumerate(self._waypoints):
            cx, cy = self._abs_to_canvas(ax, ay)
            if cx is None:
                continue
            if prev_cc:
                c.create_line(prev_cc[0], prev_cc[1], cx, cy,
                              fill=CLR_WP, width=1, dash=(4, 3),
                              tags="marker")
            rr = 7
            c.create_oval(cx-rr, cy-rr, cx+rr, cy+rr,
                          outline=CLR_WP, fill="#003300",
                          width=2, tags="marker")
            c.create_text(cx, cy, text=str(i+1),
                          fill=CLR_WP, font=("Consolas", 7, "bold"),
                          tags="marker")
            prev_cc = (cx, cy)

        # ── 레벨 ROI ──────────────────────────────────────────
        if self._level_roi:
            coords = self._roi_to_canvas(self._level_roi)
            if coords:
                x1, y1, x2, y2 = coords
                c.create_rectangle(x1, y1, x2, y2,
                                   outline=CLR_LV_ROI, width=2, dash=(6, 3),
                                   tags="marker")
                c.create_text(x1+4, y1+10,
                              text=f"레벨ROI {self._level_roi['w']}x{self._level_roi['h']}",
                              anchor=tk.W, fill=CLR_LV_ROI,
                              font=("Consolas", 8, "bold"), tags="marker")

        # ── HP ROI ────────────────────────────────────────────
        if self._hp_roi:
            coords = self._roi_to_canvas(self._hp_roi)
            if coords:
                x1, y1, x2, y2 = coords
                c.create_rectangle(x1, y1, x2, y2,
                                   outline=CLR_HP_ROI, width=2, dash=(6, 3),
                                   tags="marker")
                c.create_text(x1+4, y1+10,
                              text=f"HP ROI {self._hp_roi['w']}x{self._hp_roi['h']}",
                              anchor=tk.W, fill=CLR_HP_ROI,
                              font=("Consolas", 8, "bold"), tags="marker")

    # ─────────────────────────────────────────────────────────────
    #  실행
    # ─────────────────────────────────────────────────────────────

    def run(self):
        self._update_status()
        self._root.mainloop()


# ── 엔트리포인트 ────────────────────────────────────────────────
if __name__ == "__main__":
    app = SetupTool()
    app.run()
