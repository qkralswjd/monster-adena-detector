import torch
import numpy as np
import time
from ultralytics import YOLO

print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO('runs/detect/runs/detect/monster_v5_nano/weights/best.pt')
dummy = np.zeros((472, 1135, 3), dtype=np.uint8)

# 워밍업
print("워밍업 중...")
for _ in range(5):
    model(dummy, imgsz=320, device='cuda', verbose=False)

# 벤치마크
print("벤치마크 중...")
times = []
for _ in range(30):
    t = time.perf_counter()
    model(dummy, imgsz=320, device='cuda', verbose=False)
    times.append(time.perf_counter() - t)

avg = sum(times) / len(times)
print(f"평균: {avg*1000:.1f}ms  →  {1/avg:.1f} FPS")
print(f"최소: {min(times)*1000:.1f}ms  →  {1/min(times):.1f} FPS")
print(f"GPU 메모리: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
