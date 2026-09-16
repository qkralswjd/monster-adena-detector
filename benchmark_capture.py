import mss
import numpy as np
import time

sct = mss.mss()
mon = sct.monitors[1]

# ROI 영역
region = {
    "left":   mon["left"] + 372,
    "top":    mon["top"]  + 259,
    "width":  1135,
    "height": 472,
}

print(f"모니터: {mon['width']}x{mon['height']}")
print(f"ROI: {region['width']}x{region['height']} at ({region['left']},{region['top']})")

# 워밍업
for _ in range(10):
    sct.grab(region)

# 캡처 벤치마크
print("캡처 벤치마크 중...")
times = []
for _ in range(60):
    t = time.perf_counter()
    raw = sct.grab(region)
    frame = np.array(raw)[:, :, :3]
    times.append(time.perf_counter() - t)

avg = sum(times) / len(times)
print(f"캡처 평균: {avg*1000:.1f}ms  →  {1/avg:.1f} FPS")
print(f"캡처 최소: {min(times)*1000:.1f}ms  →  {1/min(times):.1f} FPS")

# 전체화면 캡처도 비교
region_full = mon
times2 = []
for _ in range(60):
    t = time.perf_counter()
    raw = sct.grab(region_full)
    frame = np.array(raw)[:, :, :3]
    times2.append(time.perf_counter() - t)

avg2 = sum(times2) / len(times2)
print(f"\n전체화면 평균: {avg2*1000:.1f}ms  →  {1/avg2:.1f} FPS")
print(f"\n결론: 캡처가 {'병목' if avg > 0.02 else '문제없음'} (목표 30fps = 33ms)")
