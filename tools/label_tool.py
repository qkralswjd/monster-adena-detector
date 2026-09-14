"""
label_tool.py
-------------
수집된 이미지에 몬스터 bbox를 드래그로 그려서 YOLO 포맷 라벨 저장.

사용법:
    python tools/label_tool.py

조작:
    마우스 드래그 : bbox 그리기
    Z             : 마지막 bbox 취소
    D / →         : 다음 이미지
    A / ←         : 이전 이미지
    S             : 현재 이미지 저장 (자동 저장도 됨)
    Q / ESC       : 종료

저장 위치:
    dataset/images/train/  ← 라벨링된 이미지
    dataset/labels/train/  ← YOLO 포맷 .txt
"""

import cv2
import numpy as np
import os
import glob
import sys

# ── 설정 ────────────────────────────────────────
RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset", "images", "raw")
IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset", "images", "train")
LABEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "dataset", "labels", "train")

# 클래스 정의 (필요에 따라 추가)
CLASSES = ["monster"]

# bbox 색상 (클래스별)
COLORS = [
    (0, 255, 0),    # monster → 초록
    (0, 100, 255),  # class1  → 주황
    (255, 0, 0),    # class2  → 파랑
]
# ────────────────────────────────────────────────


def ensure_dirs():
    for d in [IMG_DIR, LABEL_DIR]:
        os.makedirs(d, exist_ok=True)


def load_images():
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(RAW_DIR, ext))
    return sorted(files)


def load_labels(img_path):
    """기존 라벨 로드. 없으면 빈 리스트."""
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    boxes.append([cls_id, cx, cy, w, h])
    return boxes


def save_labels(img_path, boxes, img_w, img_h):
    """YOLO 포맷으로 라벨 저장 + 이미지 복사."""
    base = os.path.splitext(os.path.basename(img_path))[0]

    # 라벨 저장
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    with open(label_path, "w") as f:
        for box in boxes:
            cls_id, cx, cy, bw, bh = box
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # 이미지 복사 (아직 안 된 경우)
    dst_img = os.path.join(IMG_DIR, os.path.basename(img_path))
    if not os.path.exists(dst_img):
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imwrite(dst_img, img)


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """픽셀 좌표 → YOLO 정규화 좌표."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = abs(x2 - x1) / img_w
    h  = abs(y2 - y1) / img_h
    return cx, cy, w, h


def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    """YOLO 정규화 좌표 → 픽셀 좌표."""
    x1 = int((cx - bw/2) * img_w)
    y1 = int((cy - bh/2) * img_h)
    x2 = int((cx + bw/2) * img_w)
    y2 = int((cy + bh/2) * img_h)
    return x1, y1, x2, y2


def draw_frame(img, boxes, drawing, start_pt, cur_pt, cur_class, img_idx, total):
    """화면 렌더링."""
    disp = img.copy()
    h, w = img.shape[:2]

    # 저장된 bbox
    for box in boxes:
        cls_id, cx, cy, bw, bh = box
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, w, h)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        cv2.putText(disp, CLASSES[cls_id], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 드래그 중인 bbox
    if drawing and start_pt and cur_pt:
        color = COLORS[cur_class % len(COLORS)]
        cv2.rectangle(disp, start_pt, cur_pt, color, 2)

    # HUD
    labeled = "O" if boxes else "-"
    info = (f"[{img_idx+1}/{total}]  boxes:{len(boxes)}  "
            f"class:{CLASSES[cur_class]}  labeled:{labeled}")
    cv2.putText(disp, info, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.putText(disp, "Drag=bbox  Z=undo  D=next  A=prev  S=save  Q=quit",
                (5, disp.shape[0]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

    # 클래스 선택 힌트 (숫자키)
    for i, cls in enumerate(CLASSES):
        color = COLORS[i % len(COLORS)]
        mark  = ">" if i == cur_class else " "
        cv2.putText(disp, f"{mark}[{i}]{cls}", (5, 45 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return disp


def main():
    ensure_dirs()

    images = load_images()
    if not images:
        print(f"[라벨링] 이미지 없음: {RAW_DIR}")
        print("  → capture_dataset.py 먼저 실행하세요.")
        return

    print(f"[라벨링] 이미지 {len(images)}장 로드")
    print(f"[라벨링] 저장 경로: {os.path.abspath(LABEL_DIR)}\n")

    WIN = "Label Tool"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    idx       = 0
    cur_class = 0
    drawing   = False
    start_pt  = None
    cur_pt    = None

    img       = cv2.imread(images[idx])
    boxes     = load_labels(images[idx])
    saved     = True   # 현재 상태 저장 여부

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, start_pt, cur_pt, boxes, saved

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing  = True
            start_pt = (x, y)
            cur_pt   = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cur_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if drawing and start_pt:
                x1, y1 = start_pt
                x2, y2 = x, y
                if abs(x2-x1) > 5 and abs(y2-y1) > 5:
                    h, w = img.shape[:2]
                    cx, cy, bw, bh = pixel_to_yolo(x1, y1, x2, y2, w, h)
                    boxes.append([cur_class, cx, cy, bw, bh])
                    saved = False
            drawing  = False
            start_pt = None
            cur_pt   = None

    cv2.setMouseCallback(WIN, mouse_cb)

    while True:
        if img is None:
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx])
            boxes = load_labels(images[idx])
            continue

        disp = draw_frame(img, boxes, drawing, start_pt, cur_pt,
                          cur_class, idx, len(images))
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(20) & 0xFF

        # ── 종료 ──────────────────────────────
        if key in (ord('q'), 27):
            if not saved:
                h, w = img.shape[:2]
                save_labels(images[idx], boxes, w, h)
            break

        # ── 저장 ──────────────────────────────
        elif key == ord('s'):
            h, w = img.shape[:2]
            save_labels(images[idx], boxes, w, h)
            saved = True
            print(f"  [저장] {os.path.basename(images[idx])}  boxes={len(boxes)}")

        # ── undo ──────────────────────────────
        elif key == ord('z'):
            if boxes:
                boxes.pop()
                saved = False

        # ── 다음 이미지 ───────────────────────
        elif key in (ord('d'), 83):   # D or →
            if not saved:
                h, w = img.shape[:2]
                save_labels(images[idx], boxes, w, h)
                print(f"  [자동저장] {os.path.basename(images[idx])}  boxes={len(boxes)}")
            idx   = (idx + 1) % len(images)
            img   = cv2.imread(images[idx])
            boxes = load_labels(images[idx])
            saved = True

        # ── 이전 이미지 ───────────────────────
        elif key in (ord('a'), 81):   # A or ←
            if not saved:
                h, w = img.shape[:2]
                save_labels(images[idx], boxes, w, h)
                print(f"  [자동저장] {os.path.basename(images[idx])}  boxes={len(boxes)}")
            idx   = (idx - 1) % len(images)
            img   = cv2.imread(images[idx])
            boxes = load_labels(images[idx])
            saved = True

        # ── 클래스 변경 (숫자키) ──────────────
        elif ord('0') <= key <= ord('9'):
            cls = key - ord('0')
            if cls < len(CLASSES):
                cur_class = cls

    cv2.destroyAllWindows()

    # 라벨링 통계
    labeled = sum(1 for f in glob.glob(os.path.join(LABEL_DIR, "*.txt"))
                  if os.path.getsize(f) > 0)
    print(f"\n[라벨링] 완료. 라벨된 이미지: {labeled}/{len(images)}장")
    print(f"  → 다음 단계: python tools/train.py")


if __name__ == "__main__":
    main()
