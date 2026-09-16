"""
overlay_window.py
-----------------
게임화면 위에 올라가는 투명 오버레이 창.

특징:
  - 항상 게임화면 위에 표시 (topmost)
  - 배경 완전 투명 → 게임화면이 비쳐 보임
  - 탐지된 몬스터 박스 / 원 / 텍스트 표시
  - ROI 영역을 시안색 점선 박스로 표시 (전체화면 기준)
  - 오버레이 클릭 → 피코 HID로 게임에 전달
  - WS_EX_LAYERED + WS_EX_TRANSPARENT 로 클릭 통과 (표시 전용 모드)
    클릭 받는 모드는 별도 플래그로 전환

사용:
    ov = OverlayWindow(mon_left, mon_top, game_w, game_h, lb_x, ctrl,
                       roi=cfg.get("roi"))
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
CLR_ADENA_TGT = "#FF44FF"   # 아데나 탐색 중 (보라)
CLR_HUD_BG    = "#000000"   # HUD 배경
CLR_TEXT      = "#FFFFFF"   # 텍스트 흰색
CLR_TARGET_TXT= "#00FF44"
CLR_ROI       = "#00FFFF"   # ROI 테두리 (시안)

TRANSPARENT_KEY = "#010101"  # 투명 처리할 색 (완전 검정에 가까운 값)


class OverlayWindow:
    """
    게임화면 위 투명 오버레이.

    좌표계:
        오버레이 창 = 전체 모니터 크기 (mon_left, mon_top) 에서 (mon_w x mon_h)
        탐지 좌표 (det.cx, det.cy) = ROI 캡처 기준 → roi_x/roi_y 오프셋 적용
        ROI 박스 = 오버레이 위에 시안색 점선으로 표시
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
                 mon_h: int = 1080):
        """
        mon_left, mon_top : mss 모니터의 left/top (전체화면 기준)
        game_w, game_h    : 캡처 영역 크기 (ROI 사용 시 ROI 크기)
        lb_x              : 좌측 letterbox 너비 (픽셀)
        ctrl              : PicoController 또는 DummyController
        roi               : config.json 의 roi 딕셔너리 (없으면 None)
        mon_w, mon_h      : 전체 모니터 해상도 (기본 1920x1080)
        """
        self._mon_left = mon_left
        self._mon_top  = mon_top
        self._game_w   = game_w
        self._game_h   = game_h
        self._lb_x     = lb_x
        self._ctrl     = ctrl
        self._mon_w    = mon_w
        self._mon_h    = mon_h

        # ROI 설정
        self._roi = None
        if roi and roi.get("enabled", False):
            self._roi = roi  # {"x", "y", "width", "height"}

        # 오버레이 창 = 전체 모니터 크기로 확장
        self._win_x = mon_left
        self._win_y = mon_top
        self._ov_w  = mon_w
        self._ov_h  = mon_h

        # 공유 데이터 (메인 스레드 → tkinter 스레드)
        self._lock       = threading.Lock()
        self._detections = []   # Detection 객체 리스트
        self._target     = None
        self._miss_t     = 0.0
        self._det_fps    = 0.0
        self._cap_fps    = 0.0
        self._state      = "HUNTING"

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
               det_fps=0.0, cap_fps=0.0, state="HUNTING"):
        """메인루프에서 매 프레임 호출. 탐지 결과 갱신."""
        with self._lock:
            self._detections = list(detections)
            self._target     = target
            self._miss_t     = miss_elapsed
            self._det_fps    = det_fps
            self._cap_fps    = cap_fps
            self._state      = state

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

        # 오버레이 위치 = 전체 모니터 크기
        root.geometry(f"{self._ov_w}x{self._ov_h}+{self._win_x}+{self._win_y}")

        # ── 캔버스 ────────────────────────────────────────────
        self._canvas = tk.Canvas(
            root,
            width=self._ov_w,
            height=self._ov_h,
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
        event.x, event.y = 오버레이 창 기준 (= 전체 모니터 기준)
        """
        now = time.time()
        if now - self._last_click < self._click_cooldown:
            return
        self._last_click = now

        ov_x = event.x
        ov_y = event.y

        # 전체화면 기준 절대 좌표
        sc_x = self._win_x + ov_x
        sc_y = self._win_y + ov_y

        print(f"[오버레이클릭] 오버레이({ov_x},{ov_y}) → 전체화면({sc_x},{sc_y})")

        if self._ctrl and self._ctrl.is_connected:
            self._ctrl.drag_attack(sc_x, sc_y)

    # ─────────────────────────────────────────────────────────
    #  좌표 변환 헬퍼
    # ─────────────────────────────────────────────────────────

    def _det_to_ov(self, dx, dy):
        """
        탐지 좌표(ROI 기준) → 오버레이 좌표(전체 모니터 기준) 변환.
        ROI 사용 중이면 roi_x/roi_y 오프셋 적용.
        """
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

        c.delete("all")  # 전체 지우기

        with self._lock:
            dets    = list(self._detections)
            target  = self._target
            miss_t  = self._miss_t
            det_fps = self._det_fps
            cap_fps = self._cap_fps
            state   = self._state

        # ── ROI 박스 그리기 ───────────────────────────────────
        if self._roi:
            rx  = self._roi["x"]
            ry  = self._roi["y"]
            rw  = self._roi["width"]
            rh  = self._roi["height"]
            rx2 = rx + rw
            ry2 = ry + rh

            # 바깥 어둠 처리 (반투명 회색 영역 4개)
            # 위쪽
            if ry > 0:
                c.create_rectangle(0, 0, self._ov_w, ry,
                                   fill="#000000", outline="", stipple="gray25")
            # 아래쪽
            if ry2 < self._ov_h:
                c.create_rectangle(0, ry2, self._ov_w, self._ov_h,
                                   fill="#000000", outline="", stipple="gray25")
            # 왼쪽
            if rx > 0:
                c.create_rectangle(0, ry, rx, ry2,
                                   fill="#000000", outline="", stipple="gray25")
            # 오른쪽
            if rx2 < self._ov_w:
                c.create_rectangle(rx2, ry, self._ov_w, ry2,
                                   fill="#000000", outline="", stipple="gray25")

            # ROI 테두리 (시안색 점선)
            c.create_rectangle(rx, ry, rx2, ry2,
                               outline=CLR_ROI, width=2, dash=(8, 4))

            # ROI 라벨
            self._text(c, f"ROI  {rw}x{rh}  ({rx},{ry})",
                       rx + 4, ry + 16, CLR_ROI, size=9)

        # ── 탐지 박스 ─────────────────────────────────────────
        for d in dets:
            is_tgt = (target is not None and
                      d.x == target.x and d.y == target.y)

            # ROI 오프셋 적용한 오버레이 좌표
            ox,  oy  = self._det_to_ov(d.x,  d.y)
            ox2, oy2 = self._det_to_ov(d.x + d.w, d.y + d.h)
            ocx, ocy = self._det_to_ov(d.cx, d.cy)

            if is_tgt:
                continue  # 타겟은 아래서 따로

            if d.class_id == 1:  # adena
                # ADENA_CHECK 상태일 때 더 굵게 + 색 변경
                adena_clr = CLR_ADENA_TGT if state == "ADENA_CHECK" else CLR_ADENA
                adena_w   = 2             if state == "ADENA_CHECK" else 1
                c.create_rectangle(ox, oy, ox2, oy2,
                                   outline=adena_clr, width=adena_w)
                self._text(c, f"adena {d.confidence:.2f}",
                           ox, oy - 4, adena_clr, size=9)
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
            ox,  oy  = self._det_to_ov(target.x,          target.y)
            ox2, oy2 = self._det_to_ov(target.x + target.w, target.y + target.h)
            ocx, ocy = self._det_to_ov(target.cx, target.cy)

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
        state_color = CLR_ADENA_TGT if state == "ADENA_CHECK" else CLR_TARGET
        adena_count = len([d for d in dets if d.class_id == 1])
        hud = [
            f"State    : {state}",
            f"Monsters : {len([d for d in dets if d.class_id==0])}",
            f"Adenas   : {adena_count}",
            f"Det FPS  : {det_fps:.1f}",
            f"Cap FPS  : {cap_fps:.1f}",
            f"Target   : {'YES' if target else 'NONE'}",
            f"ROI      : {'ON' if self._roi else 'OFF'}",
        ]
        hud_h = len(hud) * 18 + 8
        c.create_rectangle(6, 6, 195, 6 + hud_h,
                           fill="#000000", outline="", stipple="gray50")
        for i, line in enumerate(hud):
            clr = state_color if i == 0 else CLR_TEXT
            self._text(c, line, 10, 16 + i * 18, clr, size=9)

        # ── 조작 안내 (우하단) ────────────────────────────────
        tips = ["클릭: 해당 위치 피코 클릭", "Q: 종료"]
        for i, t in enumerate(reversed(tips)):
            self._text(c, t, self._ov_w - 200,
                       self._ov_h - 12 - i * 16,
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
