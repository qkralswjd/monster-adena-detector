"""
tools/train_v2.py
-----------------
아데나 인식 개선 특화 재학습 스크립트.

현재 문제:
  - adena 학습 데이터: 64장 (monster 대비 극소수)
  - val에 adena 0장 → 학습 중 adena 성능 측정 불가
  - 클래스 불균형: monster 585개 vs adena 70개 (8:1)

해결 전략:
  1. adena 이미지를 val에 강제 포함 (stratified split)
  2. cls_pw (class weight) 로 adena 손실 가중치 8배 증가
  3. augmentation 강화 (mosaic, mixup, copy_paste)
  4. monster_v1 best.pt 에서 fine-tune (전이학습)
  5. 기존 monster_v1 모델 덮어쓰지 않고 monster_v2 로 저장

사용법:
    python tools/train_v2.py

PC에서 실행 (C:\\Users\\dongj\\monster_tracker):
    python tools/train_v2.py

결과:
    runs/detect/monster_v2/weights/best.pt
    → config.json "model" 경로를 이걸로 바꾸면 됨
"""

import os
import sys
import glob
import shutil
import random

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(ROOT, "dataset")


# ── 학습 파라미터 ────────────────────────────────────────────
BASE_MODEL  = os.path.join(ROOT, "runs", "detect", "monster_v1", "weights", "best.pt")
EPOCHS      = 80           # fine-tune은 epoch 적게
IMG_SIZE    = 640
BATCH       = 8            # RTX 2060 6GB
DEVICE      = 0
PROJECT     = os.path.join(ROOT, "runs", "detect")
NAME        = "monster_v2"

# adena 클래스 가중치: monster(1.0) vs adena(N배)
# 70개 vs 585개 → 약 8배 불균형 → 8.0 으로 보정
ADENA_CLS_WEIGHT = 8.0
# ────────────────────────────────────────────────────────────


def stratified_split(val_ratio=0.15):
    """
    adena 이미지가 반드시 val에 포함되도록 stratified split.
    - adena 이미지의 val_ratio 만큼 val로 이동
    - monster-only 이미지도 val_ratio 만큼 val로 이동
    기존 val 폴더를 정리하고 새로 구성.
    """
    train_img = os.path.join(DATASET, "images", "train")
    train_lbl = os.path.join(DATASET, "labels", "train")
    val_img   = os.path.join(DATASET, "images", "val")
    val_lbl   = os.path.join(DATASET, "labels", "val")

    # ── 기존 val → train으로 되돌리기 ──────────────────────
    print("[split] 기존 val 데이터를 train으로 복원 중...")
    for img_f in glob.glob(os.path.join(val_img, "*.jpg")):
        dst = os.path.join(train_img, os.path.basename(img_f))
        if not os.path.exists(dst):
            shutil.move(img_f, dst)
        else:
            os.remove(img_f)

    for lbl_f in glob.glob(os.path.join(val_lbl, "*.txt")):
        dst = os.path.join(train_lbl, os.path.basename(lbl_f))
        if not os.path.exists(dst):
            shutil.move(lbl_f, dst)
        else:
            os.remove(lbl_f)

    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    # ── train 이미지를 adena유/무로 분류 ───────────────────
    adena_bases   = []
    monster_bases = []

    for lbl_f in glob.glob(os.path.join(train_lbl, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl_f))[0]
        img_f = os.path.join(train_img, base + ".jpg")
        if not os.path.exists(img_f):
            continue
        with open(lbl_f) as f:
            lines = f.read().strip().split("\n")
        has_adena   = any(l.startswith("1 ") for l in lines if l)
        if has_adena:
            adena_bases.append(base)
        else:
            monster_bases.append(base)

    random.seed(42)
    random.shuffle(adena_bases)
    random.shuffle(monster_bases)

    n_val_adena   = max(1, int(len(adena_bases)   * val_ratio))
    n_val_monster = max(1, int(len(monster_bases) * val_ratio))

    val_adena   = adena_bases[:n_val_adena]
    val_monster = monster_bases[:n_val_monster]
    val_set     = val_adena + val_monster

    print(f"[split] adena   이미지: {len(adena_bases)}개  →  val: {n_val_adena}개")
    print(f"[split] monster 이미지: {len(monster_bases)}개  →  val: {n_val_monster}개")
    print(f"[split] 전체 val: {len(val_set)}개  train: {len(adena_bases)+len(monster_bases)-len(val_set)}개")

    # ── val로 이동 ──────────────────────────────────────────
    for base in val_set:
        src_img = os.path.join(train_img, base + ".jpg")
        src_lbl = os.path.join(train_lbl, base + ".txt")
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(val_img, base + ".jpg"))
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, os.path.join(val_lbl, base + ".txt"))

    # ── 결과 확인 ───────────────────────────────────────────
    val_adena_count = 0
    for lbl_f in glob.glob(os.path.join(val_lbl, "*.txt")):
        with open(lbl_f) as f:
            if any(l.startswith("1 ") for l in f.read().strip().split("\n")):
                val_adena_count += 1
    print(f"[split] val 내 adena 포함 이미지: {val_adena_count}개 ✓")

    return len(val_set)


def make_yaml():
    """dataset.yaml 생성 (경로는 상대경로로 저장)."""
    import yaml
    data = {
        "path"  : DATASET,
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : 2,
        "names" : ["monster", "adena"],
    }
    yaml_path = os.path.join(DATASET, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[yaml] {yaml_path} 생성 완료")
    return yaml_path


def train(yaml_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[train] ultralytics 없음 → pip install ultralytics")
        sys.exit(1)

    # BASE_MODEL 존재 확인
    if os.path.exists(BASE_MODEL):
        model_src = BASE_MODEL
        print(f"[train] Fine-tune 시작: {BASE_MODEL}")
    else:
        model_src = "yolov8s.pt"
        print(f"[train] monster_v1 없음 → 베이스 모델({model_src})로 처음부터 학습")

    print(f"  epochs   : {EPOCHS}")
    print(f"  img_size : {IMG_SIZE}")
    print(f"  batch    : {BATCH}")
    print(f"  adena cls_pw: {ADENA_CLS_WEIGHT}x")
    print(f"  저장     : {os.path.join(PROJECT, NAME)}\n")

    model = YOLO(model_src)
    results = model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        project   = PROJECT,
        name      = NAME,
        patience  = 20,

        # ── 클래스 불균형 보정 ──────────────────────────
        # cls_pw: 각 클래스의 BCE 손실 가중치
        # [monster_weight, adena_weight]
        cls_pw    = [1.0, ADENA_CLS_WEIGHT],

        # ── Augmentation 강화 ───────────────────────────
        mosaic    = 1.0,       # mosaic 항상 ON
        mixup     = 0.15,      # 15% 확률로 mixup
        copy_paste= 0.3,       # 30% 확률로 객체 복사-붙여넣기 (adena 증강 효과)
        degrees   = 5.0,       # 미세 회전
        translate = 0.1,
        scale     = 0.5,
        fliplr    = 0.5,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,

        # ── 기타 ────────────────────────────────────────
        save      = True,
        plots     = True,
        verbose   = True,
        close_mosaic = 10,     # 마지막 10 epoch은 mosaic OFF (fine-tuning 안정화)
    )
    return results


def print_summary():
    best_pt = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print("\n" + "=" * 55)
    print("  학습 완료!")
    print("=" * 55)
    print(f"\n  best.pt: runs/detect/{NAME}/weights/best.pt")
    print(f"\n  config.json 수정:")
    print(f'    "model": "runs/detect/{NAME}/weights/best.pt"')
    print()


def main():
    print("=" * 55)
    print("  Monster v2 학습 (아데나 인식 개선)")
    print("=" * 55)

    # 1. stratified split (adena → val 포함)
    stratified_split(val_ratio=0.15)

    # 2. dataset.yaml
    yaml_path = make_yaml()

    # 3. 학습
    train(yaml_path)

    # 4. 결과 안내
    print_summary()


if __name__ == "__main__":
    main()
