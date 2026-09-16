"""
debug_ocr.py
레벨/HP ROI 캡처 후 OCR 결과 + 이미지 저장
"""
import mss
import json
import numpy as np
import cv2
from PIL import Image

cfg = json.load(open('config.json', encoding='utf-8'))
lr = cfg['level_detector']['level_roi']
hr = cfg['level_detector']['hp_roi']

with mss.MSS() as sct:
    # 레벨 ROI
    reg = {'left': lr['x'], 'top': lr['y'], 'width': lr['w'], 'height': lr['h']}
    shot = sct.grab(reg)
    lv_img = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX')
    lv_img.save('debug_level.png')

    # HP ROI
    reg2 = {'left': hr['x'], 'top': hr['y'], 'width': hr['w'], 'height': hr['h']}
    shot2 = sct.grab(reg2)
    hp_img = Image.frombytes('RGB', shot2.size, shot2.bgra, 'raw', 'BGRX')
    hp_img.save('debug_hp.png')

# 전처리 후 저장
def preprocess(img_pil):
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    big = cv2.resize(bgr, (w*4, h*4), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

lv_bin = preprocess(lv_img)
hp_bin = preprocess(hp_img)
cv2.imwrite('debug_level_proc.png', lv_bin)
cv2.imwrite('debug_hp_proc.png', hp_bin)

print(f'레벨 ROI: {lr}')
print(f'HP ROI:  {hr}')
print()
print('이미지 저장:')
print('  debug_level.png      (원본)')
print('  debug_level_proc.png (전처리)')
print('  debug_hp.png         (원본)')
print('  debug_hp_proc.png    (전처리)')
print()

# OCR 시도
try:
    import pytesseract
    cfg_tess = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:/LEVHPlevhp"
    lv_text = pytesseract.image_to_string(lv_bin, config=cfg_tess).strip()
    hp_text = pytesseract.image_to_string(hp_bin, config=cfg_tess).strip()
    print(f'[tesseract] 레벨 OCR: "{lv_text}"')
    print(f'[tesseract] HP OCR:   "{hp_text}"')
except Exception as e:
    print(f'tesseract 없음: {e}')

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    lv_text = " ".join(reader.readtext(lv_bin, detail=0, allowlist="0123456789:/LEVHPlevhp"))
    hp_text = " ".join(reader.readtext(hp_bin, detail=0, allowlist="0123456789:/LEVHPlevhp"))
    print(f'[easyocr]   레벨 OCR: "{lv_text}"')
    print(f'[easyocr]   HP OCR:   "{hp_text}"')
except Exception as e:
    print(f'easyocr 없음: {e}')
