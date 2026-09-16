import mss
import json
import numpy as np
import cv2
from PIL import Image

cfg = json.load(open('config.json', encoding='utf-8'))
hr = cfg['level_detector']['hp_roi']

with mss.MSS() as sct:
    reg = {'left': hr['x'], 'top': hr['y'], 'width': hr['w'], 'height': hr['h']}
    shot = sct.grab(reg)
    img = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX')
    img.save('debug_hp.png')

bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

h, w = hsv.shape[:2]
mid = h // 2
print(f'HP ROI 크기: {w}x{h}')
print('중앙줄 HSV 샘플 (왼→오):')
for x in range(0, w, max(1, w//10)):
    print(f'  x={x}: HSV={hsv[mid,x]}  BGR={bgr[mid,x]}')
print('저장완료: debug_hp.png')
