"""
tools/train_field_A_v1.py
=========================
dataset_field_A_v1 (CASE A only, stride=9) 기반 YOLOv8s 학습.

설정 근거:
  - train=63장 소규모 → epochs=150, patience=20 early stop
  - CPU 환경 → batch=8, workers=2
  - pretrained=True → ImageNet weights 기반 fine-tuning
  - augmentation 적극 활용 → 소규모 데이터 다양성 보완
  - 기존 monster_v1-2 모델과 비교 가능하도록 동일 imgsz=640

출력: runs/detect/field_A_v1/
"""

from pathlib import Path
from ultralytics import YOLO
import time, json

BASE     = Path(__file__).parent.parent
DATA     = BASE / 'dataset_field_A_v1' / 'data.yaml'
PRETRAIN = 'yolov8s.pt'
PROJECT  = str(BASE / 'runs' / 'detect')
NAME     = 'field_A_v1'

def main():
    print("=" * 65)
    print("  🚀 YOLOv8s 학습 시작 — dataset_field_A_v1 (CASE A only)")
    print(f"  data   : {DATA}")
    print(f"  model  : {PRETRAIN}")
    print(f"  output : {PROJECT}/{NAME}")
    print("=" * 65)

    model = YOLO(PRETRAIN)

    t0 = time.time()
    results = model.train(
        data      = str(DATA),
        epochs    = 150,
        patience  = 20,          # early stopping
        imgsz     = 640,
        batch     = 8,
        workers   = 2,
        device    = 'cpu',
        project   = PROJECT,
        name      = NAME,
        exist_ok  = False,

        # 옵티마이저
        optimizer = 'AdamW',
        lr0       = 0.001,
        lrf       = 0.01,        # 최종 lr = lr0 * lrf
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 5.0,

        # 소규모 데이터 augmentation
        mosaic    = 1.0,
        mixup     = 0.1,
        fliplr    = 0.5,
        flipud    = 0.1,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        copy_paste = 0.1,        # 소규모에서 bbox 다양성 보완

        # 저장
        save      = True,
        save_period = 10,        # 10 epoch마다 체크포인트
        plots     = True,
        val       = True,
        verbose   = True,
    )

    elapsed = time.time() - t0
    print(f"\n  학습 완료: {elapsed/60:.1f}분")

    # 결과 저장
    out_dir = Path(PROJECT) / NAME
    metrics = {
        'elapsed_min': round(elapsed/60, 1),
        'best_map50':  float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'best_map5095': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        'best_precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'best_recall':    float(results.results_dict.get('metrics/recall(B)', 0)),
    }
    with open(out_dir / 'train_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  📊 최종 결과")
    print(f"     mAP@50    : {metrics['best_map50']:.4f}")
    print(f"     mAP@50-95 : {metrics['best_map5095']:.4f}")
    print(f"     Precision  : {metrics['best_precision']:.4f}")
    print(f"     Recall     : {metrics['best_recall']:.4f}")
    print(f"\n  best.pt: {out_dir/'weights'/'best.pt'}")
    print("=" * 65)

if __name__ == '__main__':
    main()
