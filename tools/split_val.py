"""
split_val.py
------------
train 데이터에서 20%를 랜덤으로 val로 이동.
이미지와 라벨 파일을 함께 이동.

사용:
  python tools/split_val.py
"""

import os
import random
import shutil

BASE    = os.path.dirname(os.path.dirname(__file__))
TR_IMG  = os.path.join(BASE, "dataset", "images", "train")
VAL_IMG = os.path.join(BASE, "dataset", "images", "val")
TR_LBL  = os.path.join(BASE, "dataset", "labels", "train")
VAL_LBL = os.path.join(BASE, "dataset", "labels", "val")

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# train 이미지 목록
exts = (".jpg", ".jpeg", ".png")
images = [f for f in os.listdir(TR_IMG) if f.lower().endswith(exts)]

if not images:
    print("[오류] train 이미지가 없어요!")
    exit()

# 20% 랜덤 선택
random.seed(42)
random.shuffle(images)
val_count = max(1, int(len(images) * 0.2))
val_images = images[:val_count]

print(f"전체: {len(images)}장  →  train: {len(images)-val_count}장  val: {val_count}장")
print("이동 중...")

moved = 0
no_label = 0

for img_file in val_images:
    # 이미지 이동
    src_img = os.path.join(TR_IMG, img_file)
    dst_img = os.path.join(VAL_IMG, img_file)
    shutil.move(src_img, dst_img)

    # 라벨 이동 (확장자만 .txt로 변경)
    name     = os.path.splitext(img_file)[0]
    src_lbl  = os.path.join(TR_LBL, name + ".txt")
    dst_lbl  = os.path.join(VAL_LBL, name + ".txt")
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
        moved += 1
    else:
        no_label += 1
        print(f"  [경고] 라벨 없음: {name}.txt")

print(f"\n완료!")
print(f"  이동된 이미지+라벨: {moved}장")
if no_label:
    print(f"  라벨 없는 이미지:   {no_label}장 (이미지만 이동됨)")
print(f"\n  train: {len(images)-val_count}장")
print(f"  val:   {val_count}장")
