"""
main.py
-------
Monster Tracker - 피코 HID 공격 통합 버전

전체 흐름:
  1. 화면 캡처 (mss monitors[1], left=-1920)
  2. YOLOv8s 탐지 (monster=0 / adena=1)
  3. ROI 필터링
  4. ROI 안 몬스터 전체 목록 로그 출력
  5. 탐지 순서대로 타겟 큐 구성 (1번→2번→...)
  6. IoU 매칭으로 타겟 추적
  7. 피코로 드래그 공격 (PRESS→아래드래그→RELEASE)
  8. 사망/소실(1.5초) → 다음 타겟으로

좌표 기준:
  프레임좌표  = mss 캡처 기준 (0~1920, 0~1080) ← 피코 MOVE 계산 기준
  전체화면좌표 = 프레임좌표 + mon_left/top
                 게임이 left=-1920 이면 전체화면x = 프레임x - 1920 (음수, 정상)

피코 좌표 계산:
  커서 리셋(MOVE:-9999:-9999) 후 게임 중앙으로 이동
  → 이후 프레임좌표 = 피코 추적 좌표 1:1 대응
  dx = target.cx - _cur_x → MOVE:{dx*SCALE}:{dy*SCALE}

핫키:
  Q / ESC : 종료 (창 모드)
  C       : 타겟 수동 해제
  Ctrl+C  : 강제종료
"""

import sys
import os
import time
import json
from typing import Optional, List

sys.path.insert(0, os.path.dirname(__file__))

from screen_capture import ScreenCapture
from detector import YOLODetector, Detection
from target_selector import find_matching
from death_detector import DeathDetector
from controller import PicoController, DummyController


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MonsterTrackerApp:

    WIN = "Monster Tracker"

    def __init__(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        self._cfg = load_config(cfg_path)

        # ── 화면 캡처 ───────────────────────────────
        self._capture = ScreenCapture(self._cfg["capture"]["monitor"])

        # ── YOLO 탐지기 ─────────────────────────────
        dcfg = self._cfg["detector"]
        self._detector = YOLODetector(
            model_path    = dcfg["model"],
            confidence    = dcfg["confidence"],
            iou_threshold = dcfg["iou_threshold"],
            device        = dcfg["device"],
            img_size      = dcfg["img_size"],
            classes       = dcfg["classes"],
        )

        # ── 사망 감지기 ─────────────────────────────
        tcfg = self._cfg["target"]
        self._death_detector = DeathDetector(
            timeout_sec = tcfg["death_timeout_sec"],
            iou_thresh  = tcfg["death_iou_thresh"],
        )

        # ── 모니터 정보 ─────────────────────────────
        mon = self._capture._monitor
        self._mon_left = mon["left"]   # 게임모니터 = -1920
        self._mon_top  = mon["top"]    # 보통 0
        mss_w = mon["width"]
        mss_h = mon["height"]
        print(f"[Capture] 게임모니터: left={self._mon_left}, top={self._mon_top} "
              f"| 캡처 크기: {mss_w}x{mss_h}")
        print(f"[Capture] 전체화면좌표 = 프레임좌표 + ({self._mon_left},{self._mon_top})")
        print(f"[Capture] 피코 MOVE 기준 = 프레임좌표 (음수 전체화면좌표 아님)")

        # ── 레터박스 보정 ────────────────────────────
        lcfg = self._cfg.get("letterbox", {})
        if lcfg.get("auto_detect", True):
            self._letterbox_x = self._detect_letterbox()
        else:
            self._letterbox_x = lcfg.get("x", 0)
        if self._letterbox_x > 0:
            print(f"[Letterbox] 좌측 검은 여백: {self._letterbox_x}px → 클릭 X좌표 보정 적용")
        else:
            print(f"[Letterbox] 검은 여백 없음 (보정 불필요)")

        # ── ROI (탐지 필터 존) ──────────────────────
        rcfg = self._cfg["roi"]
        if rcfg["width"] > 0 and rcfg["height"] > 0:
            self._roi = (rcfg["x"], rcfg["y"], rcfg["width"], rcfg["height"])
            print(f"[ROI] 탐지존: 프레임({rcfg['x']},{rcfg['y']}) "
                  f"크기 {rcfg['width']}x{rcfg['height']}")
        else:
            self._roi = None
            print("[ROI] 전체 화면 탐지")

        # ── 피코 컨트롤러 ────────────────────────────
        ccfg = self._cfg["controller"]
        gcfg = self._cfg["game"]
        if ccfg["enabled"]:
            self._ctrl = PicoController(
                port      = ccfg["port"],
                baudrate  = ccfg["baudrate"],
                screen_w  = gcfg["width"],
                screen_h  = gcfg["height"],
                mon_left  = self._mon_left,   # 전체화면 기준 (-1920)
                mon_top   = self._mon_top,    # 전체화면 기준 (0)
            )
            if not self._ctrl.connect():
                print("[Controller] 연결 실패 → DummyController로 대체")
                self._ctrl = DummyController()
                self._ctrl.connect()
        else:
            print("[Controller] enabled=false → DummyController")
            self._ctrl = DummyController()
            self._ctrl.connect()

        # ── 공격 설정 ────────────────────────────────
        acfg = self._cfg["attack"]
        self._attack_enabled  = acfg["enabled"]
        self._attack_cooldown = acfg["cooldown_sec"]
        self._drag_dx         = acfg.get("drag_dx", 0)
        self._drag_dy         = acfg.get("drag_dy", 30)
        self._hold_ms         = acfg.get("hold_ms", 80)
        self._aim_offset_y    = acfg.get("aim_offset_y", 0)
        self._click_offset_x  = acfg.get("click_offset_x", 0)
        self._click_offset_y  = acfg.get("click_offset_y", 0)
        self._last_attack_time = 0.0
        self._attacked_target_no = -1  # 이미 공격한 타겟 번호 (타겟당 1회만 클릭)
        print(f"[Attack] enabled={self._attack_enabled} "
              f"drag=({self._drag_dx},{self._drag_dy}) "
              f"hold={self._hold_ms}ms "
              f"offset=({self._click_offset_x},{self._click_offset_y})")

        # ── 타겟 상태 ───────────────────────────────
        self._target: Optional[Detection] = None
        self._target_no: int = 0
        self._running = False

        # ── FPS 제한 ────────────────────────────────
        self._cap_fps        = self._cfg["capture"]["fps"]
        self._frame_interval = 1.0 / self._cap_fps

        # ── 로그 타이머 ─────────────────────────────
        self._last_status_time = 0.0
        self._last_coord_time  = 0.0
        self._last_monsters_log: List = []  # 직전 프레임 (cx,cy) 목록

    # ── 레터박스 자동 감지 ────────────────────────────
    def _detect_letterbox(self) -> int:
        """
        게임 화면 캡처 후 좌측 검은 여백 너비 자동 감지.
        화면 중앙 Y라인에서 검은 픽셀(R+G+B<30) 끝나는 X 반환.
        """
        import numpy as np
        import cv2
        frame = self._capture.capture_full()
        if frame is None:
            return 0
        h, w = frame.shape[:2]
        mid_y = h // 2
        row = frame[mid_y]
        for x in range(w):
            b, g, r = int(row[x][0]), int(row[x][1]), int(row[x][2])
            if r + g + b > 30:
                return x
        return 0

    # ── 전체화면 절대좌표 변환 (로그용) ───────────────
    def _to_screen(self, frame_x: int, frame_y: int):
        return frame_x + self._mon_left, frame_y + self._mon_top

    # ══════════════════════════════════════════════
    #  몬스터 목록 로그 (변화 있을 때만 출력)
    # ══════════════════════════════════════════════
    def _log_monsters(self, monsters: List[Detection]):
        cur_ids = [(m.cx, m.cy) for m in monsters]
        if cur_ids == self._last_monsters_log:
            return
        self._last_monsters_log = cur_ids

        if not monsters:
            print(f"[탐지] 몬스터 없음")
            return

        print(f"[탐지] 몬스터 {len(monsters)}마리 ─────────────────")
        for i, m in enumerate(monsters, 1):
            sc_x, sc_y = self._to_screen(m.cx, m.cy)
            target_mark = " ◀ 현재타겟" if (self._target and
                          m.cx == self._target.cx and
                          m.cy == self._target.cy) else ""
            print(f"  [{i}번] 프레임({m.cx:4d},{m.cy:4d})  "
                  f"전체화면({sc_x:5d},{sc_y:4d})  "
                  f"conf={m.confidence:.2f}  "
                  f"bbox=({m.x},{m.y},{m.w},{m.h}){target_mark}")

    # ══════════════════════════════════════════════
    #  타겟 갱신 (탐지 순서대로 1→2→3...)
    # ══════════════════════════════════════════════
    def _update_target(self, monsters: List[Detection]) -> bool:
        """
        타겟 없음 → 탐지 순서 1번(리스트 첫 번째)을 타겟으로 지정.
        타겟 있음 → IoU 매칭으로 추적.
        소실 1.5초 → 사망 확정 → True 반환.
        """
        if self._target is None:
            if monsters:
                self._target = monsters[0]
                self._target_no += 1
                self._attacked_target_no = -1  # 새 타겟 선택 시 초기화 → 반드시 공격
                self._death_detector.reset()
                sc_x, sc_y = self._to_screen(self._target.cx, self._target.cy)
                print(f"[Target #{self._target_no}] 선택: "
                      f"프레임({self._target.cx},{self._target.cy})  "
                      f"전체화면({sc_x},{sc_y})  "
                      f"conf={self._target.confidence:.2f}")
            return False

        # IoU 매칭으로 같은 몬스터 추적
        matched = find_matching(monsters, self._target,
                                self._cfg["target"]["death_iou_thresh"])
        if matched:
            self._target = matched
            self._death_detector.reset()
            return False

        # 소실 → 사망 판정
        dead = self._death_detector.update(monsters, self._target)
        if dead:
            sc_x, sc_y = self._to_screen(self._target.cx, self._target.cy)
            print(f"[Target #{self._target_no}] 사망/소실 확정! "
                  f"마지막위치: 프레임({self._target.cx},{self._target.cy})  "
                  f"전체화면({sc_x},{sc_y})")
            return True

        return False

    # ══════════════════════════════════════════════
    #  피코 공격 실행
    # ══════════════════════════════════════════════
    def _do_attack(self):
        """
        타겟이 있고 쿨타임이 지나면 피코로 드래그 공격.

        ★ 좌표 기준: 전체화면 좌표 (sc_x, sc_y) 를 피코에 전달!
          피코 리셋(0,0) = Windows(0,0) = 전체화면(0,0) 기준
          게임이 left=-1920 → sc_x 음수 → MOVE:음수 → 왼쪽 이동 → 게임 화면 도달

        aim_offset_y 적용:
          타겟 cx/cy는 바운딩박스 중심 → aim_offset_y=-10 이면 10px 위(머리)를 조준
          공격 모션: PRESS → 아래로 drag_dy만큼 드래그 → RELEASE
        """
        if not self._attack_enabled:
            print("[Attack] SKIP: attack_enabled=False")
            return
        if self._target is None:
            return  # 타겟 없음은 정상 상황이므로 로그 생략

        # 이미 이 타겟을 공격했으면 스킵 (타겟당 1회만 클릭)
        if self._target_no == self._attacked_target_no:
            return  # 이미 공격 완료 - 정상 상황이므로 로그 생략

        # 쿨타임 체크
        now = time.time()
        elapsed_since_last = now - self._last_attack_time
        if elapsed_since_last < self._attack_cooldown:
            print(f"[Attack] SKIP: 쿨타임 중 ({elapsed_since_last:.2f}s < {self._attack_cooldown}s)")
            return

        # 공격 중이면 스킵 (단, 스레드 타임아웃 2초 초과 시 강제 해제)
        is_atk = getattr(self._ctrl, 'is_attacking', False)
        if is_atk:
            if now - self._last_attack_time > 2.0:
                self._ctrl._attacking = False  # 2초 넘으면 강제 해제
                print(f"[Attack] is_attacking 강제 해제 (2초 초과)")
            else:
                print(f"[Attack] SKIP: 공격 스레드 실행 중")
                return

        # 전체화면 좌표 계산 → 피코에 전달
        # letterbox 보정: YOLO cx는 검은여백 포함 기준 → 여백만큼 빼면 게임 내 실제 X
        frame_x = self._target.cx + self._click_offset_x - self._letterbox_x
        frame_y = self._target.cy + self._aim_offset_y + self._click_offset_y
        sc_x, sc_y = self._to_screen(frame_x, frame_y)

        print(f"[Attack #{self._target_no}] 공격! "
              f"프레임({frame_x},{frame_y})  전체화면({sc_x},{sc_y})  drag_dy={self._drag_dy}")

        self._ctrl.drag_attack(
            x       = sc_x,   # 전체화면 좌표 (음수 가능) - SCALE 곱해 MOVE 전송
            y       = sc_y,
            drag_dx = self._drag_dx,
            drag_dy = self._drag_dy,
            hold_ms = self._hold_ms,
        )
        self._last_attack_time = time.time()
        self._attacked_target_no = self._target_no  # 이 타겟은 공격 완료

    # ══════════════════════════════════════════════
    #  현재 타겟 이동 좌표 로그 (0.1초마다)
    # ══════════════════════════════════════════════
    def _log_target_coord(self):
        if self._target is None:
            return
        now = time.time()
        if now - self._last_coord_time < 0.1:
            return
        self._last_coord_time = now

        sc_x, sc_y = self._to_screen(self._target.cx, self._target.cy)
        miss = self._death_detector.miss_elapsed
        miss_str = f"  소실중={miss:.1f}s" if miss > 0 else ""
        atk_str  = " [공격중]" if (hasattr(self._ctrl, 'is_attacking') and
                                    self._ctrl.is_attacking) else ""
        print(f"[타겟 #{self._target_no}] "
              f"프레임({self._target.cx:4d},{self._target.cy:4d})  "
              f"전체화면({sc_x:5d},{sc_y:4d})  "
              f"conf={self._target.confidence:.2f}{miss_str}{atk_str}")

    # ══════════════════════════════════════════════
    #  메인 루프
    # ══════════════════════════════════════════════
    def run(self):
        show_window = self._cfg.get("display", {}).get("show_window", False)
        self._running = True

        if show_window:
            import cv2
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN, 960, 540)
            print("[MonsterTracker] 창 모드. Q/ESC=종료  C=타겟해제")
        else:
            print("[MonsterTracker] 백그라운드 모드. Ctrl+C=종료")

        print("=" * 60)
        last_time = time.time()

        try:
            while self._running:
                # ── FPS 제한 ──────────────────────────
                now = time.time()
                elapsed = now - last_time
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                last_time = time.time()

                # ── 캡처 ──────────────────────────────
                frame = self._capture.capture()
                if frame is None:
                    continue

                # ── 탐지 ──────────────────────────────
                all_detections = self._detector.detect(frame)

                # ── ROI 필터링 ────────────────────────
                if self._roi:
                    rx, ry, rw, rh = self._roi
                    all_detections = [
                        d for d in all_detections
                        if rx <= d.cx <= rx + rw and ry <= d.cy <= ry + rh
                    ]

                monsters = [d for d in all_detections if d.class_id == 0]
                adenas   = [d for d in all_detections if d.class_id == 1]

                # ── 몬스터 전체 목록 로그 ─────────────
                self._log_monsters(monsters)

                # ── 타겟 갱신 ─────────────────────────
                dead = self._update_target(monsters)
                if dead:
                    self._target = None
                    self._attacked_target_no = -1  # 다음 타겟 공격 가능하도록 초기화

                # ── 피코 공격 ─────────────────────────
                self._do_attack()

                # ── 현재 타겟 좌표 출력 ───────────────
                self._log_target_coord()

                # ── 5초마다 상태 요약 ─────────────────
                if time.time() - self._last_status_time > 5.0:
                    ctrl_status = ("연결" if self._ctrl.is_connected else "끊김")
                    target_str = (f"#{self._target_no} 프레임({self._target.cx},{self._target.cy})"
                                  if self._target else "없음")
                    print(f"[Status] monster={len(monsters)} | "
                          f"adena={len(adenas)} | "
                          f"타겟={target_str} | "
                          f"피코={ctrl_status} | "
                          f"탐지FPS={self._detector.fps}")
                    self._last_status_time = time.time()

                # ── 창 모드 시각화 ────────────────────
                if show_window:
                    self._render_window(frame, all_detections)

        except KeyboardInterrupt:
            print("\n[MonsterTracker] Ctrl+C → 종료")
        finally:
            self._ctrl.stop()
            self._ctrl.disconnect()
            if show_window:
                import cv2
                cv2.destroyAllWindows()
            print("[MonsterTracker] 종료 완료")

    # ══════════════════════════════════════════════
    #  창 모드 렌더링
    # ══════════════════════════════════════════════
    def _render_window(self, frame, all_detections):
        try:
            import cv2
            import visualizer
            miss = self._death_detector.miss_elapsed
            out = visualizer.draw(
                frame        = frame,
                detections   = all_detections,
                target       = self._target,
                miss_elapsed = miss,
                detector_fps = self._detector.fps,
                capture_fps  = self._capture.fps,
                roi          = self._roi,
            )
            cv2.imshow(self.WIN, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._running = False
            elif key == ord("c"):
                self._target = None
                self._death_detector.reset()
                self._last_monsters_log = []
                print("[Target] 수동 해제")
        except Exception as e:
            print(f"[Render] 오류: {e}")


def set_roi_interactive():
    """
    실시간 캡처 화면에서 마우스 드래그로 ROI 선택 → config.json 저장.
    python main.py --set-roi 로 실행.
    """
    import mss
    import cv2
    import numpy as np

    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    cfg = load_config(cfg_path)

    mon_idx = cfg["capture"]["monitor"]
    sct = mss.mss()
    mon = sct.monitors[mon_idx]
    cap_w, cap_h = mon["width"], mon["height"]
    disp_w, disp_h = cap_w // 2, cap_h // 2

    roi_data = {"start": None, "end": None, "dragging": False}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_data["start"]    = (x, y)
            roi_data["end"]      = (x, y)
            roi_data["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and roi_data["dragging"]:
            roi_data["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_data["end"]      = (x, y)
            roi_data["dragging"] = False

    WIN = "ROI 설정 (드래그 → Enter저장 / R초기화 / ESC취소)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("\n[ROI 설정] 실시간 캡처 화면에서 탐지 영역을 드래그하세요.")
    print("  드래그 = ROI 선택")
    print("  Enter  = 저장")
    print("  R      = 선택 초기화")
    print("  ESC    = 취소\n")

    while True:
        # 실시간 캡처
        shot  = sct.grab(mon)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img   = cv2.resize(frame, (disp_w, disp_h))

        # 기존 ROI 표시 (노란색)
        rcfg = cfg["roi"]
        if rcfg["width"] > 0:
            rx = rcfg["x"] // 2
            ry = rcfg["y"] // 2
            rw = rcfg["width"] // 2
            rh = rcfg["height"] // 2
            cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
            cv2.putText(img, f"현재ROI ({rcfg['x']},{rcfg['y']}) {rcfg['width']}x{rcfg['height']}",
                        (rx+4, ry+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 드래그 선택 영역 표시 (초록색)
        if roi_data["start"] and roi_data["end"]:
            s, e = roi_data["start"], roi_data["end"]
            cv2.rectangle(img, s, e, (0, 255, 0), 2)
            fx = min(s[0], e[0]) * 2
            fy = min(s[1], e[1]) * 2
            fw = abs(e[0] - s[0]) * 2
            fh = abs(e[1] - s[1]) * 2
            label = f"({fx},{fy}) {fw}x{fh}"
            cv2.putText(img, label,
                        (min(s[0], e[0]) + 4, min(s[1], e[1]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 안내 텍스트
        cv2.putText(img, "드래그:선택  Enter:저장  R:초기화  ESC:취소",
                    (5, disp_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)

        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC - 취소
            print("[ROI] 취소")
            break

        elif key == ord('r') or key == ord('R'):  # R - 초기화
            roi_data["start"]    = None
            roi_data["end"]      = None
            roi_data["dragging"] = False
            print("[ROI] 선택 초기화")

        elif key in (13, 32):  # Enter / Space - 저장
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
                    print(f"[ROI] 저장 완료: 프레임({fx},{fy}) 크기 {fw}x{fh}")
                    print(f"  config.json 업데이트 완료")
                    break
                else:
                    print("[ROI] 너무 작습니다. 다시 드래그하세요.")
            else:
                print("[ROI] 먼저 드래그로 영역을 선택하세요.")

    sct.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--set-roi" in sys.argv:
        set_roi_interactive()
    else:
        app = MonsterTrackerApp()
        app.run()
