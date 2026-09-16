import mss

with mss.mss() as sct:
    for i, m in enumerate(sct.monitors):
        print(f"monitors[{i}]: left={m['left']} top={m['top']} {m['width']}x{m['height']}")
    print(f"\n게임이 실행되는 모니터 번호를 config.json의 'monitor'에 설정하세요.")
