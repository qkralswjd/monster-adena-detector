"""
overlay_window.py
-----------------
게임화면 위에 올라가는 투명 오버레이 창.

특징:
  - 항상 게임화면 위에 표시 (topmost)
  - 배경 완전 투명 → 게임화면이 비쳐 보임
  - 탐지된 몬스터 박스 / 원 / 텍스트 표시
  - 오버레이 클릭 → 피코 HID로 게임에 전달
  - WS_EX_LAYERED + WS_EX_TRANSPARENT 로 클릭 통과 (표시 전용 모드)
    클릭 받는 모드는 별도 플래그로 전환

사용:
    ov = OverlayWindow(mon_left, mon_top, game_w, game_h, lb_x, ctrl)
    ov.update(detections)   # 매 프레임 호출
    ov.start()              # 별도 스레드에서 tkinter 루프 실행
    ov.stop()
"""

import tkinter as tk
import threading
import time
import math
from typing import List, Optional


# ── 색상 (tkinter hex) ──────────────────────────────────────
CLR_BOX       = "#FF6600"   # 몬스터 박스 (주황)
CLR_TARGET    = "#00FF44"   # 현재 타겟 (초록)
CLR_DEAD      = "#FF2222"   # 소실/사망 (빨강)
CLR_ADENA     = "#FFD700"   # 아데나 (금색)
CLR_HUD_BG    = "#000000"   # HUD 배경
CLR_TEXT      = "#FFFFFF"   # 텍스트 흰색
CLR_TARGET_TXT= "#00FF44"

TRANSPARENT_KEY = "#010101"  # 투명 처리할 색 (완전 검정에 가까운 값)


class OverlayWindow:
    """
    게임화면 위 투명 오버레이.

    좌표계:
        오버레이 창 = 게임 영역 (mon_left+lb_x, mon_top) 에서 (game_w x game_h)
        탐지 좌표 (det.cx, det.cy) = 1920x1080 프레임 기준
        → 오버레이 상의 좌표 = det.cx - lb_x, det.cy
    """

    def __init__(self,
                 mon_left: int,
                 mon_top: int,
                 game_w: int,
                 game_h: int,
                 lb_x: int,
                 ctrl=None):
        """
        mon_left, mon_top : mss 모니터의 left/top (전체화면 기준)
        game_w, game_h    : 실제 게임 영역 크기 (letterbox 제외)
        lb_x              : 좌측 letterbox 너비 (픽셀)
        ctrl              : PicoController 또는 DummyController
        """
        self._mon_left = mon_left
        self._mon_top  = mon_top
        self._game_w   = game_w
        self._game_h   = game_h
        self._lb_x     = lb_x
        self._ctrl     = ctrl

        # 오버레이 창의 Windows 절대 좌표
        self._win_x = mon_left + lb_x
        self._win_y = mon_top

        # 공유 데이터 (메인 스레드 → tkinter 스레드)
        self._lock       = threading.Lock()
        self._detections = []   # Detection 객체 리스트
        self._target     = None
        self._miss_t     = 0.0
        self._det_fps    = 0.0
        self._cap_fps    = 0.0

        self._running    = False
        self._root       = None
        self._canvas     = None
        self._thread     = None

        # 클릭 쿨다운
        self._last_click = 0.0
        self._click_cooldown = 0.3  # 초

    # ─────────────────────────────────────────────────────────
    #  외부 API
    # ─────────────────────────────────────────────────────────

    def update(self, detections, target=None, miss_elapsed=0.0,
               det_fps=0.0, cap_fps=0.0):
        """메인루프에서 매 프레임 호출. 탐지 결과 갱신."""
        with self._lock:
            self._detections = list(detections)
            self._target     = target
            self._miss_t     = miss_elapsed
            self._det_fps    = det_fps
            self._cap_fps    = cap_fps

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
    #  tkinter 루프 (별도 스레드)
    # ─────────────────────────────────────────────────────────

    def _tk_loop(self):
        self._root = tk.Tk()
        root = self._root

        # ── 창 설정 ───────────────────────────────────────────
        root.overrideredirect(True)          # 타이틀바 제거
        root.attributes("-topmost", True)    # 항상 위
        root.attributes("-transparentcolor", TRANSPARENT_KEY)  # 투명색 설정
        root.attributes("-alpha", 1.0)
        root.configure(bg=TRANSPARENT_KEY)

        # 오버레이 위치 = 게임 영역 좌상단
        root.geometry(f"{self._game_w}x{self._game_h}+{self._win_x}+{self._win_y}")

        # ── 캔버스 ────────────────────────────────────────────
        self._canvas = tk.Canvas(
            root,
            width=self._game_w,
            height=self._game_h,
            bg=TRANSPARENT_KEY,
            highlightthickness=0
        )
        self._canvas.pack()

        # ── 마우스 클릭 → 피코 전달 ──────────────────────────
        self._canvas.bind("<Button-1>", self._on_click)

        # ── 주기적 갱신 ───────────────────────────────────────
        self._schedule_redraw()

        root.mainloop()

    def _schedule_redraw(self):
        if not self._running:
            return
        self._redraw()
        if self._root:
            self._root.after(33, self._schedule_redraw)  # ~30fps

    # ─────────────────────────────────────────────────────────
    #  클릭 처리
    # ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        """
        오버레이 캔버스 클릭 → 게임 좌표로 변환 → 피코 전달.
        event.x, event.y = 오버레이 창 기준 (= 게임 영역 기준)
        """
        now = time.time()
        if now - self._last_click < self._click_cooldown:
            return
        self._last_click = now

        # 오버레이 좌표 → Windows 전체화면 좌표
        # 오버레이(0,0) = 게임영역 좌상단 = Windows(mon_left+lb_x, mon_top)
        ov_x = event.x
        ov_y = event.y

        # 전체화면 기준 절대 좌표
        sc_x = self._win_x + ov_x   # = mon_left + lb_x + ov_x
        sc_y = self._win_y + ov_y   # = mon_top  + ov_y

        print(f"[오버레이클릭] 오버레이({ov_x},{ov_y}) → 전체화면({sc_x},{sc_y})")

        if self._ctrl and self._ctrl.is_connected:
            self._ctrl.drag_attack(sc_x, sc_y)

    # ─────────────────────────────────────────────────────────
    #  그리기
    # ─────────────────────────────────────────────────────────

    def _redraw(self):
        c = self._canvas
        if c is None:
            return

        c.delete("all")  # 전체 지우기

        with self._lock:
            dets    = list(self._detections)
            target  = self._target
            miss_t  = self._miss_t
            det_fps = self._det_fps
            cap_fps = self._cap_fps

        # ── 탐지 박스 ─────────────────────────────────────────
        for d in dets:
            is_tgt = (target is not None and
                      d.x == target.x and d.y == target.y)

            # 오버레이 좌표 = 프레임 좌표 - lb_x
            ox  = d.x  - self._lb_x
            oy  = d.y
            ox2 = d.x + d.w - self._lb_x
            oy2 = d.y + d.h
            ocx = d.cx - self._lb_x
            ocy = d.cy

            if is_tgt:
                continue  # 타겟은 아래서 따로

            if d.class_id == 1:  # adena
                c.create_rectangle(ox, oy, ox2, oy2,
                                   outline=CLR_ADENA, width=1)
                self._text(c, f"adena {d.confidence:.2f}",
                           ox, oy - 4, CLR_ADENA, size=9)
            else:  # monster
                c.create_rectangle(ox, oy, ox2, oy2,
                                   outline=CLR_BOX, width=1)
                self._text(c, f"{d.confidence:.2f}",
                           ox, oy - 4, CLR_BOX, size=9)
                # 중심점
                c.create_oval(ocx-3, ocy-3, ocx+3, ocy+3,
                              fill=CLR_BOX, outline="")

        # ── 타겟 박스 ─────────────────────────────────────────
        if target is not None:
            ox  = target.x  - self._lb_x
            oy  = target.y
            ox2 = target.x + target.w - self._lb_x
            oy2 = target.y + target.h
            ocx = target.cx - self._lb_x
            ocy = target.cy

            color = CLR_DEAD if miss_t > 0 else CLR_TARGET
            label = (f"MISSING {miss_t:.1f}s" if miss_t > 0
                     else f"TARGET  {target.confidence:.2f}")
            thick = 1 if miss_t > 0 else 2

            # 박스
            c.create_rectangle(ox, oy, ox2, oy2,
                               outline=color, width=thick)

            # 코너 강조
            cs = 14
            for px, py, dx, dy in [
                (ox,  oy,  cs,  cs),
                (ox2, oy,  -cs, cs),
                (ox,  oy2, cs,  -cs),
                (ox2, oy2, -cs, -cs),
            ]:
                c.create_line(px, py, px+dx, py, fill=color, width=3)
                c.create_line(px, py, px, py+dy, fill=color, width=3)

            # 중심 원
            r = 6
            c.create_oval(ocx-r, ocy-r, ocx+r, ocy+r,
                          fill=color, outline="")

            # 라벨
            self._text(c, label, ox, oy - 10, color, size=10)

        # ── HUD (좌상단) ──────────────────────────────────────
        hud = [
            f"Monsters : {len([d for d in dets if d.class_id==0])}",
            f"Det FPS  : {det_fps:.1f}",
            f"Cap FPS  : {cap_fps:.1f}",
            f"Target   : {'YES' if target else 'NONE'}",
            f"LB offset: {self._lb_x}px",
        ]
        # 반투명 HUD 배경
        hud_h = len(hud) * 18 + 8
        c.create_rectangle(6, 6, 180, 6 + hud_h,
                           fill="#000000", outline="", stipple="gray50")
        for i, line in enumerate(hud):
            self._text(c, line, 10, 16 + i * 18, CLR_TEXT, size=9)

        # ── 조작 안내 (우하단) ────────────────────────────────
        tips = ["클릭: 해당 위치 피코 클릭", "Q: 종료"]
        for i, t in enumerate(reversed(tips)):
            self._text(c, t, self._game_w - 200,
                       self._game_h - 12 - i * 16,
                       "#AAAAAA", size=8)

    @staticmethod
    def _text(canvas, txt, x, y, color, size=10):
        """그림자 효과 텍스트."""
        canvas.create_text(x+1, y+1, text=txt, anchor="sw",
                           fill="#000000",
                           font=("Consolas", size, "bold"))
        canvas.create_text(x, y, text=txt, anchor="sw",
                           fill=color,
                           font=("Consolas", size, "bold"))
