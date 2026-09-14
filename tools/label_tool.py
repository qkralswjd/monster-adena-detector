"""
label_tool.py
-------------
클릭 한 번으로 자동 bbox 감지 + 수동 드래그 수정 지원.

동작:
    1. 이미지에서 몬스터 위치 클릭
    2. GrabCut으로 자동 bbox 계산
    3. 맘에 안 들면 드래그로 직접 그리기
    4. D키 → 다음 이미지 (자동 저장)

조작:
    클릭        : 해당 위치 자동 bbox
    드래그      : 수동 bbox 직접 그리기
    Z           : 마지막 bbox 취소
    D / →       : 다음 이미지 (자동저장)
    A / ←       : 이전 이미지 (자동저장)
    S           : 수동 저장
    F           : 현재 이미지 스킵 (라벨 없이 넘김)
    Q / ESC     : 종료

저장:
    dataset/images/train/
    dataset/labels/train/  ← YOLO 포맷 .txt
"""

import cv2
import numpy as np
import os
import glob
import shutil

# ── 설정 ────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR   = os.path.join(ROOT, "dataset", "images", "raw")
IMG_DIR   = os.path.join(ROOT, "dataset", "images", "train")
LABEL_DIR = os.path.join(ROOT, "dataset", "labels", "train")
CLASSES   = ["monster"]

# 자동 bbox 크기 (클릭 주변 탐색 범위)
AUTO_SEARCH_RADIUS = 80   # 클릭 주변 이 범위 안에서 외곽선 찾기
MIN_AREA           = 800  # 너무 작은 건 무시
# ────────────────────────────────────────────────

CLR_BOX    = (0, 255, 0)
CLR_DRAG   = (0, 200, 255)
CLR_TEXT   = (255, 255, 255)
CLR_SHADOW = (0, 0, 0)


def ensure_dirs():
    for d in [IMG_DIR, LABEL_DIR]:
        os.makedirs(d, exist_ok=True)


def load_images():
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files += glob.glob(os.path.join(RAW_DIR, ext))
    return sorted(files)


def load_labels(img_path, img_w, img_h):
    base = os.path.splitext(os.path.basename(img_path))[0]
    path = os.path.join(LABEL_DIR, base + ".txt")
    boxes = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    # 픽셀로 변환해서 저장
                    x1 = int((cx - bw/2) * img_w)
                    y1 = int((cy - bh/2) * img_h)
                    x2 = int((cx + bw/2) * img_w)
                    y2 = int((cy + bh/2) * img_h)
                    boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def save_labels(img_path, boxes, img_w, img_h):
    base     = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    img_dst  = os.path.join(IMG_DIR, os.path.basename(img_path))

    # 라벨 저장
    with open(lbl_path, "w") as f:
        for cls_id, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = abs(x2 - x1) / img_w
            bh = abs(y2 - y1) / img_h
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # 이미지 복사
    if not os.path.exists(img_dst):
        shutil.copy2(img_path, img_dst)


def auto_bbox(img, click_x, click_y):
    """
    클릭 위치 주변에서 자동으로 bbox 계산.
    1차: 외곽선(contour) 탐지
    2차: 안 되면 클릭 중심으로 고정 크기 박스
    """
    h, w = img.shape[:2]

    # 클릭 주변 ROI 잘라내기
    rx1 = max(0, click_x - AUTO_SEARCH_RADIUS)
    ry1 = max(0, click_y - AUTO_SEARCH_RADIUS)
    rx2 = min(w, click_x + AUTO_SEARCH_RADIUS)
    ry2 = min(h, click_y + AUTO_SEARCH_RADIUS)
    roi = img[ry1:ry2, rx1:rx2]

    # 그레이 + 블러 + 엣지
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_box  = None
    best_dist = float('inf')
    lx = click_x - rx1  # ROI 내 클릭 좌표
    ly = click_y - ry1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # 클릭 위치와 박스 중심 거리
        bcx = bx + bw // 2
        bcy = by + bh // 2
        dist = ((bcx - lx)**2 + (bcy - ly)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            # 전체 이미지 좌표로 변환
            best_box = (rx1 + bx, ry1 + by,
                        rx1 + bx + bw, ry1 + by + bh)

    if best_box:
        return best_box

    # fallback: 클릭 중심으로 고정 크기
    fw, fh = 80, 100
    return (max(0, click_x - fw//2), max(0, click_y - fh//2),
            min(w, click_x + fw//2), min(h, click_y + fh//2))


def put_text(img, txt, x, y, color=CLR_TEXT, scale=0.55, thick=1):
    cv2.putText(img, txt, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, CLR_SHADOW, thick+1, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_frame(img, boxes, drag_start, drag_cur, idx, total, img_path):
    disp = img.copy()
    h, w = img.shape[:2]

    # 저장된 박스들
    for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(disp, (x1, y1), (x2, y2), CLR_BOX, 2)
        put_text(disp, f"{CLASSES[cls_id]}", x1, y1 - 5,
                 color=CLR_BOX, scale=0.45)

    # 드래그 중인 박스
    if drag_start and drag_cur:
        cv2.rectangle(disp, drag_start, drag_cur, CLR_DRAG, 2)

    # 상단 HUD
    fname   = os.path.basename(img_path)
    labeled = len(boxes) > 0
    put_text(disp, f"[{idx+1}/{total}] {fname}  boxes:{len(boxes)}",
             8, 22, scale=0.55)
    put_text(disp, "클릭=자동bbox  드래그=수동  Z=취소  D=다음  A=이전  F=스킵  Q=종료",
             8, h - 8, scale=0.42, color=(180, 180, 180))

    # 라벨 상태 표시
    status_color = (0, 255, 0) if labeled else (0, 0, 255)
    status_txt   = "LABELED" if labeled else "NO LABEL"
    put_text(disp, status_txt, w - 110, 22,
             color=status_color, scale=0.55, thick=1)

    return disp


class State:
    def __init__(self):
        self.drag_start = None
        self.drag_cur   = None
        self.is_drag    = False
        self.moved      = False   # 드래그인지 클릭인지 구분


def main():
    ensure_dirs()

    images = load_images()
    if not images:
        print(f"[라벨링] 이미지 없음: {RAW_DIR}")
        print("  → capture_dataset.py 먼저 실행하세요.")
        return

    print(f"[라벨링] 이미지 {len(images)}장")
    print(f"[라벨링] 클릭=자동bbox  드래그=수동  D=다음  A=이전  Z=취소  F=스킵  Q=종료\n")

    WIN = "Label Tool"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    idx   = 0
    state = State()

    img   = cv2.imread(images[idx])
    ih, iw = img.shape[:2]
    boxes  = load_labels(images[idx], iw, ih)
    saved  = True

    def mouse_cb(event, x, y, flags, param):
        nonlocal boxes, saved

        if event == cv2.EVENT_LBUTTONDOWN:
            state.drag_start = (x, y)
            state.drag_cur   = (x, y)
            state.is_drag    = True
            state.moved      = False

        elif event == cv2.EVENT_MOUSEMOVE and state.is_drag:
            state.drag_cur = (x, y)
            dx = abs(x - state.drag_start[0])
            dy = abs(y - state.drag_start[1])
            if dx > 8 or dy > 8:
                state.moved = True

        elif event == cv2.EVENT_LBUTTONUP and state.is_drag:
            state.is_drag = False

            if not state.moved:
                # ── 클릭 → 자동 bbox ──────────────
                box = auto_bbox(img, x, y)
                if box:
                    boxes.append((0, *box))
                    saved = False
            else:
                # ── 드래그 → 수동 bbox ─────────────
                x1, y1 = state.drag_start
                x2, y2 = x, y
                if abs(x2-x1) > 10 and abs(y2-y1) > 10:
                    boxes.append((0,
                                  min(x1,x2), min(y1,y2),
                                  max(x1,x2), max(y1,y2)))
                    saved = False

            state.drag_start = None
            state.drag_cur   = None

    cv2.setMouseCallback(WIN, mouse_cb)

    def go_next():
        nonlocal idx, img, boxes, saved, ih, iw
        if not saved:
            save_labels(images[idx], boxes, iw, ih)
            print(f"  [저장] {os.path.basename(images[idx])}  boxes={len(boxes)}")
        idx   = (idx + 1) % len(images)
        img   = cv2.imread(images[idx])
        ih, iw = img.shape[:2]
        boxes  = load_labels(images[idx], iw, ih)
        saved  = True

    def go_prev():
        nonlocal idx, img, boxes, saved, ih, iw
        if not saved:
            save_labels(images[idx], boxes, iw, ih)
            print(f"  [저장] {os.path.basename(images[idx])}  boxes={len(boxes)}")
        idx   = (idx - 1) % len(images)
        img   = cv2.imread(images[idx])
        ih, iw = img.shape[:2]
        boxes  = load_labels(images[idx], iw, ih)
        saved  = True

    while True:
        disp = draw_frame(img, boxes,
                          state.drag_start if state.moved else None,
                          state.drag_cur   if state.moved else None,
                          idx, len(images), images[idx])
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):          # Q/ESC: 종료
            if not saved:
                save_labels(images[idx], boxes, iw, ih)
            break

        elif key == ord('s'):              # S: 저장
            save_labels(images[idx], boxes, iw, ih)
            saved = True
            print(f"  [저장] {os.path.basename(images[idx])}  boxes={len(boxes)}")

        elif key == ord('z'):              # Z: undo
            if boxes:
                boxes.pop()
                saved = False

        elif key == ord('f'):              # F: 스킵 (라벨 없이 넘김)
            print(f"  [스킵] {os.path.basename(images[idx])}")
            idx   = (idx + 1) % len(images)
            img   = cv2.imread(images[idx])
            ih, iw = img.shape[:2]
            boxes  = load_labels(images[idx], iw, ih)
            saved  = True

        elif key in (ord('d'), 83, 13):    # D / → / Enter: 다음
            go_next()

        elif key in (ord('a'), 81):        # A / ←: 이전
            go_prev()

    cv2.destroyAllWindows()

    labeled = sum(
        1 for f in glob.glob(os.path.join(LABEL_DIR, "*.txt"))
        if os.path.getsize(f) > 0
    )
    print(f"\n[라벨링] 완료. {labeled}/{len(images)}장 라벨됨")
    print("  → 다음: python tools/train.py")


if __name__ == "__main__":
    main()
