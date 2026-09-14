"""
main.py
-------
Monster Tracker - 몬스터 탐지 + 피코 공격

흐름:
  1. 화면 캡처 (mss)
  2. YOLO 탐지 (monster=0, adena=1)
  3. ROI 필터링
  4. 타겟 선택 (첫 번째 몬스터)
  5. IoU 추적 → 1.5초 소실 시 사망
  6. 피코 드래그 공격 (타겟당 1회)

핫키 (show_window=true):
  Q / ESC : 종료
  C       : 타겟 수동 해제
  R       : ROI 재설정 (실시간 드래그)
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection
from controller import PicoController, DummyController

import cv2
import numpy as np


# ══════════════════════════════════════════════
#  IoU 계산
# ══════════════════════════════════════════════
def calc_iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def find_match(detections, target, iou_thresh=0.3):
    """IoU 매칭으로 같은 몬스터 찾기"""
    best, best_iou = None, iou_thresh
    for d in detections:
        iou = calc_iou(d, target)
        if iou > best_iou:
            best, best_iou = d, iou
    return best


# ══════════════════════════════════════════════
#  ROI 실시간 설정
# ══════════════════════════════════════════════
def set_roi_interactive(cfg, cfg_path):
    """실시간 캡처 화면에서 드래그로 ROI 설정 → config.json 저장"""
    import mss
    sct = mss.mss()
    mon_idx = cfg["capture"]["monitor"]
    mon = sct.monitors[mon_idx]
    cap_w, cap_h = mon["width"], mon["height"]
    disp_w, disp_h = cap_w // 2, cap_h // 2

    roi_data = {"start": None, "end": None, "dragging": False}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_data["start"] = (x, y)
            roi_data["end"]   = (x, y)
            roi_data["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and roi_data["dragging"]:
            roi_data["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_data["end"]      = (x, y)
            roi_data["dragging"] = False

    WIN = "ROI 설정 (드래그→Enter저장 / R초기화 / ESC취소)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("\n[ROI] 드래그로 탐지 영역 선택 → Enter저장 / R초기화 / ESC취소")

    while True:
        shot  = sct.grab(mon)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img   = cv2.resize(frame, (disp_w, disp_h))

        # 기존 ROI (노란색)
        rcfg = cfg["roi"]
        if rcfg["width"] > 0:
            rx, ry = rcfg["x"] // 2, rcfg["y"] // 2
            rw, rh = rcfg["width"] // 2, rcfg["height"] // 2
            cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
            cv2.putText(img, f"현재ROI ({rcfg['x']},{rcfg['y']}) {rcfg['width']}x{rcfg['height']}",
                        (rx+4, ry+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 드래그 선택 (초록색)
        if roi_data["start"] and roi_data["end"]:
            s, e = roi_data["start"], roi_data["end"]
            cv2.rectangle(img, s, e, (0, 255, 0), 2)
            fx = min(s[0], e[0]) * 2
            fy = min(s[1], e[1]) * 2
            fw = abs(e[0] - s[0]) * 2
            fh = abs(e[1] - s[1]) * 2
            cv2.putText(img, f"({fx},{fy}) {fw}x{fh}",
                        (min(s[0],e[0])+4, min(s[1],e[1])+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(img, "드래그:선택  Enter:저장  R:초기화  ESC:취소",
                    (5, disp_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            print("[ROI] 취소")
            break
        elif key in (ord('r'), ord('R')):
            roi_data["start"] = roi_data["end"] = None
            roi_data["dragging"] = False
            print("[ROI] 초기화")
        elif key in (13, 32):  # Enter / Space
            if roi_data["start"] and roi_data["end"]:
                s, e = roi_data["start"], roi_data["end"]
                fx = min(s[0], e[0]) * 2
                fy = min(s[1], e[1]) * 2
                fw = abs(e[0] - s[0]) * 2
                fh = abs(e[1] - s[1]) * 2
                if fw > 20 and fh > 20:
                    cfg["roi"]["x"]      = fx
                    cfg["roi"]["y"]      = fy
                    cfg["roi"]["width"]  = fw
                    cfg["roi"]["height"] = fh
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=4, ensure_ascii=False)
                    print(f"[ROI] 저장: ({fx},{fy}) {fw}x{fh}")
                    break
                else:
                    print("[ROI] 너무 작음. 다시 드래그하세요.")
            else:
                print("[ROI] 먼저 드래그하세요.")

    sct.close()
    cv2.destroyWindow(WIN)
    return cfg


# ══════════════════════════════════════════════
#  메인 앱
# ══════════════════════════════════════════════
class App:

    WIN = "Monster Tracker"

    def __init__(self, cfg: dict):
        self._cfg = cfg

        # 캡처
        self._cap = ScreenCapture(cfg["capture"]["monitor"])
        mon = self._cap._monitor
        self._mon_left = mon["left"]
        self._mon_top  = mon["top"]
        print(f"[Capture] 모니터: left={self._mon_left} top={self._mon_top} "
              f"{mon['width']}x{mon['height']}")

        # 탐지기
        d = cfg["detector"]
        self._det = YOLODetector(
            model_path    = d["model"],
            confidence    = d["confidence"],
            iou_threshold = d["iou_threshold"],
            device        = d["device"],
            img_size      = d["img_size"],
        )

        # ROI
        r = cfg["roi"]
        self._roi = (r["x"], r["y"], r["width"], r["height"]) if r["width"] > 0 else None

        # 피코 컨트롤러
        c = cfg["controller"]
        if c["enabled"]:
            self._ctrl = PicoController(
                port     = c["port"],
                baudrate = c["baudrate"],
                mon_left = self._mon_left,
                mon_top  = self._mon_top,
            )
            if not self._ctrl.connect():
                print("[Controller] 연결 실패 → Dummy 사용")
                self._ctrl = DummyController()
                self._ctrl.connect()
        else:
            self._ctrl = DummyController()
            self._ctrl.connect()

        # 공격 설정
        a = cfg["attack"]
        self._atk_enabled  = a["enabled"]
        self._atk_cooldown = a["cooldown_sec"]
        self._drag_dx      = a.get("drag_dx", 0)
        self._drag_dy      = a.get("drag_dy", 30)
        self._hold_ms      = a.get("hold_ms", 80)
        self._offset_x     = a.get("click_offset_x", 0)
        self._offset_y     = a.get("click_offset_y", 0)
        self._aim_y        = a.get("aim_offset_y", 0)
        print(f"[Attack] enabled={self._atk_enabled}  "
              f"drag=({self._drag_dx},{self._drag_dy})  hold={self._hold_ms}ms")

        # 타겟 상태
        self._target       = None   # 현재 타겟 Detection
        self._target_no    = 0      # 타겟 번호 (새 타겟마다 +1)
        self._attacked_no  = -1     # 이미 공격한 타겟 번호
        self._miss_since   = None   # 소실 시작 시각
        self._death_sec    = cfg["target"]["death_timeout_sec"]
        self._iou_thresh   = cfg["target"]["death_iou_thresh"]
        self._last_atk_t   = 0.0
        self._running      = False

        # FPS
        self._interval = 1.0 / cfg["capture"]["fps"]

    # ── 전체화면 좌표 변환 ────────────────────────
    def _to_screen(self, fx, fy):
        return fx + self._mon_left, fy + self._mon_top

    # ── 타겟 갱신 ─────────────────────────────────
    def _update_target(self, monsters):
        """
        반환: True = 사망확정 (타겟 교체 필요)
        """
        if self._target is None:
            if monsters:
                self._target     = monsters[0]
                self._target_no += 1
                self._attacked_no = -1
                self._miss_since  = None
                sc = self._to_screen(self._target.cx, self._target.cy)
                print(f"[Target #{self._target_no}] 선택: "
                      f"프레임({self._target.cx},{self._target.cy})  "
                      f"전체화면({sc[0]},{sc[1]})")
            return False

        # IoU 매칭
        matched = find_match(monsters, self._target, self._iou_thresh)
        if matched:
            self._target     = matched
            self._miss_since = None
            return False

        # 소실 타이머
        now = time.time()
        if self._miss_since is None:
            self._miss_since = now
        if now - self._miss_since >= self._death_sec:
            print(f"[Target #{self._target_no}] 사망/소실 확정")
            self._miss_since = None
            return True

        return False

    # ── 공격 ──────────────────────────────────────
    def _attack(self):
        if not self._atk_enabled or self._target is None:
            return

        # 이미 이 타겟 공격했으면 스킵
        if self._target_no == self._attacked_no:
            return

        # 쿨타임
        now = time.time()
        if now - self._last_atk_t < self._atk_cooldown:
            return

        # 공격 중이면 스킵 (2초 초과 시 강제 해제)
        if self._ctrl.is_attacking:
            if now - self._last_atk_t > 2.0:
                self._ctrl._attacking = False
            else:
                return

        fx = self._target.cx + self._offset_x
        fy = self._target.cy + self._aim_y + self._offset_y
        sc_x, sc_y = self._to_screen(fx, fy)

        print(f"[Attack #{self._target_no}] "
              f"프레임({fx},{fy})  전체화면({sc_x},{sc_y})")

        self._ctrl.drag_attack(
            sc_x    = sc_x,
            sc_y    = sc_y,
            drag_dx = self._drag_dx,
            drag_dy = self._drag_dy,
            hold_ms = self._hold_ms,
        )
        self._last_atk_t  = time.time()
        self._attacked_no = self._target_no

    # ── 메인 루프 ─────────────────────────────────
    def run(self):
        show = self._cfg.get("display", {}).get("show_window", True)
        self._running = True

        if show:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN, 960, 540)
            print("[MonsterTracker] Q/ESC=종료  C=타겟해제  R=ROI재설정")
        else:
            print("[MonsterTracker] 백그라운드 모드. Ctrl+C=종료")
        print("=" * 50)

        last_t      = time.time()
        last_log_t  = 0.0

        try:
            while self._running:
                # FPS 제한
                now = time.time()
                diff = now - last_t
                if diff < self._interval:
                    time.sleep(self._interval - diff)
                last_t = time.time()

                # 캡처
                frame = self._cap.capture()
                if frame is None:
                    continue

                # 탐지
                dets = self._det.detect(frame)

                # ROI 필터
                if self._roi:
                    rx, ry, rw, rh = self._roi
                    dets = [d for d in dets
                            if rx <= d.cx <= rx+rw and ry <= d.cy <= ry+rh]

                monsters = [d for d in dets if d.class_id == 0]

                # 타겟 갱신
                dead = self._update_target(monsters)
                if dead:
                    self._target      = None
                    self._attacked_no = -1

                # 공격
                self._attack()

                # 타겟 좌표 로그 (0.1초마다)
                now = time.time()
                if self._target and now - last_log_t >= 0.1:
                    last_log_t = now
                    sc = self._to_screen(self._target.cx, self._target.cy)
                    miss = (now - self._miss_since) if self._miss_since else 0
                    miss_s = f"  소실중={miss:.1f}s" if miss > 0 else ""
                    print(f"[타겟 #{self._target_no}] "
                          f"프레임({self._target.cx},{self._target.cy})  "
                          f"전체화면({sc[0]},{sc[1]})  "
                          f"conf={self._target.confidence:.2f}{miss_s}")

                # 창 표시
                if show:
                    self._draw(frame, dets, monsters)

        except KeyboardInterrupt:
            print("\n[MonsterTracker] 종료")
        finally:
            self._ctrl.stop()
            self._ctrl.disconnect()
            if show:
                cv2.destroyAllWindows()

    # ── 시각화 ────────────────────────────────────
    def _draw(self, frame, dets, monsters):
        img = cv2.resize(frame, (960, 540))
        sx, sy = 960 / frame.shape[1], 540 / frame.shape[0]

        # ROI
        if self._roi:
            rx, ry, rw, rh = self._roi
            cv2.rectangle(img,
                          (int(rx*sx), int(ry*sy)),
                          (int((rx+rw)*sx), int((ry+rh)*sy)),
                          (0, 255, 255), 1)

        # 탐지 박스
        for d in dets:
            x1 = int(d.x * sx); y1 = int(d.y * sy)
            x2 = int((d.x+d.w) * sx); y2 = int((d.y+d.h) * sy)
            color = (0, 80, 255) if d.class_id == 0 else (0, 200, 100)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(img, label, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 타겟 강조
        if self._target:
            cx = int(self._target.cx * sx)
            cy = int(self._target.cy * sy)
            cv2.circle(img, (cx, cy), 12, (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 3,  (0, 0, 255), -1)
            cv2.putText(img, f"TARGET #{self._target_no}",
                        (cx+14, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 상단 HUD
        cv2.putText(img, f"monster={len(monsters)}  fps={self._det.fps}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow(self.WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self._running = False
        elif key == ord('c'):
            self._target      = None
            self._attacked_no = -1
            self._miss_since  = None
            print("[Target] 수동 해제")
        elif key == ord('r'):
            self._ctrl.stop()
            cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
            self._cfg = set_roi_interactive(self._cfg, cfg_path)
            r = self._cfg["roi"]
            self._roi = (r["x"], r["y"], r["width"], r["height"]) if r["width"] > 0 else None
            print(f"[ROI] 갱신: {self._roi}")


# ══════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if "--set-roi" in sys.argv:
        cfg = set_roi_interactive(cfg, cfg_path)
    else:
        App(cfg).run()
