import mss

with mss.MSS() as sct:
    for i, m in enumerate(sct.monitors):
        print(f"monitors[{i}]: {m['width']}x{m['height']}  left={m['left']} top={m['top']}")
