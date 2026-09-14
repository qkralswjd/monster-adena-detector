"""
train.py
--------
YOLOv8s 커스텀 학습 실행.
라벨링 완료 후 실행하면 dataset을 train/val로 자동 분할하고 학습 시작.

사용법:
    python tools/train.py

결과:
    runs/detect/monster_v1/weights/best.pt  ← 최종 모델
    → config.json "model" 에 이 경로 입력
"""

import os
import sys
import glob
import shutil
import random
import yaml

# ── 설정 ────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET    = os.path.join(ROOT, "dataset")
CLASSES    = ["monster", "adena"]  # 클래스 이름 (label_tool.py와 동일하게)

# 학습 파라미터 (RTX 2060 기준)
MODEL      = "yolov8s.pt"        # 베이스 모델
EPOCHS     = 100
IMG_SIZE   = 640
BATCH      = 8                   # RTX 2060 6GB → 8 정도가 안전
DEVICE     = 0                   # GPU 0번 (cuda:0)
VAL_RATIO  = 0.2                 # 검증셋 비율 20%
PROJECT    = os.path.join(ROOT, "runs", "detect")
NAME       = "monster_v1"
# ────────────────────────────────────────────────


def split_dataset():
    """train 이미지를 train/val 로 분할."""
    src_img_dir   = os.path.join(DATASET, "images", "train")
    src_lbl_dir   = os.path.join(DATASET, "labels", "train")
    val_img_dir   = os.path.join(DATASET, "images", "val")
    val_lbl_dir   = os.path.join(DATASET, "labels", "val")

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # 라벨 있는 이미지만
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in glob.glob(os.path.join(src_img_dir, ext)):
            base     = os.path.splitext(os.path.basename(f))[0]
            lbl_path = os.path.join(src_lbl_dir, base + ".txt")
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                images.append(base)

    if not images:
        print("[train] 라벨된 이미지 없음. label_tool.py 먼저 실행하세요.")
        sys.exit(1)

    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_RATIO))
    val_set   = images[:n_val]
    train_set = images[n_val:]

    print(f"[train] 전체: {len(images)}장  →  train: {len(train_set)}, val: {len(val_set)}")

    # val 복사
    for base in val_set:
        for ext in [".jpg", ".jpeg", ".png"]:
            src = os.path.join(src_img_dir, base + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(val_img_dir, base + ext))
                break
        lbl_src = os.path.join(src_lbl_dir, base + ".txt")
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(val_lbl_dir, base + ".txt"))

    return len(train_set), len(val_set)


def make_yaml():
    """dataset.yaml 생성."""
    data = {
        "path"  : DATASET,
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(CLASSES),
        "names" : CLASSES,
    }
    yaml_path = os.path.join(DATASET, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[train] dataset.yaml 생성: {yaml_path}")
    return yaml_path


def train(yaml_path):
    """YOLOv8 학습 실행."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[train] ultralytics 없음 → pip install ultralytics")
        sys.exit(1)

    print(f"\n[train] 학습 시작")
    print(f"  모델    : {MODEL}")
    print(f"  epochs  : {EPOCHS}")
    print(f"  img_size: {IMG_SIZE}")
    print(f"  batch   : {BATCH}")
    print(f"  device  : cuda:{DEVICE}")
    print(f"  저장    : {os.path.join(PROJECT, NAME)}\n")

    model = YOLO(MODEL)
    results = model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        project   = PROJECT,
        name      = NAME,
        patience  = 20,          # 20 epoch 개선 없으면 early stop
        save      = True,
        plots     = True,
        verbose   = True,
    )
    return results


def main():
    print("=" * 50)
    print("  Monster YOLOv8s 학습")
    print("=" * 50)

    # 1. train/val 분할
    n_train, n_val = split_dataset()

    # 2. dataset.yaml 생성
    yaml_path = make_yaml()

    # 3. 학습
    results = train(yaml_path)

    # 4. 결과 안내
    best_pt = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print("\n" + "=" * 50)
    print("  학습 완료!")
    print("=" * 50)
    print(f"\n  best.pt 경로:")
    print(f"  {os.path.abspath(best_pt)}")
    print(f"\n  config.json 수정:")
    print(f'  "model": "{os.path.abspath(best_pt).replace(chr(92), "/")}"')
    print()


if __name__ == "__main__":
    main()
