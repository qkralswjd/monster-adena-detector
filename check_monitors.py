import mss
from PIL import Image

with mss.MSS() as sct:
    print(f"총 모니터 수: {len(sct.monitors)-1}개")
    print()
    for i, mon in enumerate(sct.monitors):
        print(f"monitors[{i}]: left={mon['left']} top={mon['top']} width={mon['width']} height={mon['height']}")
        if i == 0:
            print("         (전체 합성 모니터 - 스킵)")
            continue
        # 각 모니터 캡처 후 저장
        shot = sct.grab(mon)
        img = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX')
        fname = f'monitor_{i}.png'
        img.save(fname)
        print(f"         → 저장: {fname}")
