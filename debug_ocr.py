"""debug_ocr.py
레벨 OCR + HP 바 진단 스크립트 (레퍼런스 debug_hp_bar.py + debug_level_ocr.py 통합)

실행하면:
  1. config.json 에서 level_roi / hp_roi 읽기
  2. 화면 캡처 후 해당 영역 크롭
  3. 이미지 파일 저장 (원본 / 마스크 / overlay / 전처리)
  4. HP 바 열별 픽셀 분석 출력
  5. easyocr + tesseract 레벨 OCR 결과 출력

저장 파일:
  debug_level_crop.png      — 레벨 원본 크롭
  debug_level_processed.png — 레벨 전처리 (2배 확대 + CLAHE + OTSU)
  debug_hp_crop.png         — HP 원본 크롭
  debug_hp_mask.png         — HP HSV 빨간색 마스크 (빨간=감지)
  debug_hp_overlay.png      — HP 원본 + 마스크 overlay

Windows 실행:
  cd C:\\Users\\dongj\\monster_tracker
  python debug_ocr.py
"""

import json
import sys
import cv2
import mss
import numpy as np
from PIL import Image

# ── config 로드 ──────────────────────────────────────────────────────────────
cfg = json.load(open('config.json', encoding='utf-8'))

monitor_index = cfg.get('capture', {}).get('monitor', 1)
ldcfg = cfg.get('level_detector', {})
level_roi = ldcfg.get('level_roi')   # {x, y, w, h}
hp_roi    = ldcfg.get('hp_roi')      # {x, y, w, h}

print(f"모니터 인덱스  : {monitor_index}")
print(f"레벨 ROI       : {level_roi}")
print(f"HP ROI         : {hp_roi}")

if level_roi is None and hp_roi is None:
    print("\n⚠️  level_roi / hp_roi 가 모두 null 입니다.")
    print("   → setup_tool.py 를 실행해서 ROI 를 먼저 등록하세요.")
    sys.exit(1)

# ── 화면 캡처 ────────────────────────────────────────────────────────────────
with mss.mss() as sct:
    monitors = sct.monitors
    print(f"\n=== 모니터 목록 ===")
    for i, m in enumerate(monitors):
        print(f"  [{i}] {m}")

    if monitor_index >= len(monitors):
        print(f"ERROR: monitor_index={monitor_index} 없음. 최대={len(monitors)-1}")
        sys.exit(1)

    mon = monitors[monitor_index]
    print(f"\n사용 모니터: {mon}")

    shot = sct.grab(mon)
    frame = np.array(shot)[:, :, :3]          # BGRA → BGR (3채널)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

print(f"frame size: {frame.shape[1]}x{frame.shape[0]} (w×h)")


# ── 헬퍼: ROI dict 키 통일 (w/width, h/height 둘 다 지원) ───────────────────
def _parse_roi(roi: dict):
    x = roi.get('x', 0)
    y = roi.get('y', 0)
    w = roi.get('w', roi.get('width', 0))
    h = roi.get('h', roi.get('height', 0))
    return x, y, w, h


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HP 바 진단 (레퍼런스 debug_hp_bar.py 구조)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if hp_roi is not None:
    print("\n" + "="*60)
    print("  HP 바 진단")
    print("="*60)

    x, y, w, h = _parse_roi(hp_roi)
    print(f"HP 크롭 좌표: x={x}, y={y}, w={w}, h={h}")

    fh, fw = frame.shape[:2]
    x2 = min(x + w, fw)
    y2 = min(y + h, fh)

    if x < 0 or y < 0 or x >= fw or y >= fh:
        print(f"ERROR: HP ROI 가 프레임 밖입니다! frame=({fw},{fh})")
    else:
        crop_hp = frame[y:y2, x:x2]
        print(f"crop shape: {crop_hp.shape}")

        # HSV 빨간색 마스크 (레퍼런스 hp_reader.py _HP_HSV_RANGES)
        HP_HSV_RANGES = [
            ((0,  80, 40), (10,  255, 255)),   # 빨간색 영역 1
            ((170, 80, 40), (180, 255, 255)),   # 빨간색 영역 2 (wrap-around)
        ]
        hsv = cv2.cvtColor(crop_hp, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in HP_HSV_RANGES:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, m)

        col_red_count = np.sum(mask > 0, axis=0)
        col_has_red   = col_red_count > 0
        total_cols    = mask.shape[1]
        total_rows    = mask.shape[0]
        total_red     = int(np.count_nonzero(mask))

        # 연속 채워진 열 (레퍼런스 debug_hp_bar.py 분석용 — 참고값)
        filled_cols = 0
        for has_red in col_has_red:
            if has_red:
                filled_cols += 1
            else:
                break

        # 실제 HP% 계산: 전체 빨간 열 합계 비율 (레퍼런스 hp_reader.py _calc_hp_pct)
        red_cols_total = int(np.count_nonzero(col_has_red))
        hp_pct_main  = round((red_cols_total / total_cols) * 100.0, 1)
        hp_pct_seq   = round(filled_cols / total_cols * 100.0, 1)
        hp_pct_pixel = round(total_red / (total_cols * total_rows) * 100.0, 1)

        print(f"\n=== HP 분석 결과 ===")
        print(f"전체 열 수          : {total_cols}")
        print(f"전체 행 수          : {total_rows}")
        print(f"전체 빨간 픽셀      : {total_red} / {total_cols * total_rows}")
        print(f"빨간 열 수(전체)    : {red_cols_total} / {total_cols}")
        print(f"연속 채워진 열      : {filled_cols} / {total_cols}")
        print(f"★ HP% (전체열합계) : {hp_pct_main}%   ← level_detector.py 실제 사용값")
        print(f"  HP% (연속열)      : {hp_pct_seq}%   ← 참고 (텍스트 끊김으로 부정확)")
        print(f"  HP% (픽셀비율)    : {hp_pct_pixel}%   ← 참고")

        # 처음 30열 분석
        print(f"\n=== 처음 30열 빨간 픽셀 수 ===")
        for i in range(min(30, total_cols)):
            bar = "█" * min(col_red_count[i], 20)
            print(f"  col[{i:3d}]: {col_red_count[i]:3d}px  {bar}")

        # 마지막 10열
        print(f"\n=== 마지막 10열 빨간 픽셀 수 ===")
        for i in range(max(0, total_cols - 10), total_cols):
            bar = "█" * min(col_red_count[i], 20)
            print(f"  col[{i:3d}]: {col_red_count[i]:3d}px  {bar}")

        # 이미지 저장
        cv2.imwrite('debug_hp_crop.png', crop_hp)
        print(f"\n✅ debug_hp_crop.png 저장 완료")

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[mask > 0] = (0, 0, 255)
        cv2.imwrite('debug_hp_mask.png', mask_color)
        print(f"✅ debug_hp_mask.png 저장 완료")

        overlay = crop_hp.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.imwrite('debug_hp_overlay.png', overlay)
        print(f"✅ debug_hp_overlay.png 저장 완료")

        print(f"\n→ debug_hp_crop.png 확인: HP 바가 올바르게 잡혔는지 확인")
        print(f"→ debug_hp_mask.png 확인: 빨간 픽셀이 어디에 있는지 확인")
        if hp_pct_main < 1.0:
            print(f"\n⚠️  HP% 가 {hp_pct_main}% 로 매우 낮습니다.")
            print(f"   → debug_hp_crop.png 를 확인해서 HP 바 ROI 가 올바른지 점검하세요.")
            print(f"   → HP 바가 파란색이면 HSV 범위 조정이 필요합니다.")

else:
    print("\n⚠️  hp_roi 가 null — HP 진단 건너뜀")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  레벨 OCR 진단 (레퍼런스 debug_level_ocr.py 구조)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if level_roi is not None:
    print("\n" + "="*60)
    print("  레벨 OCR 진단")
    print("="*60)

    x, y, w, h = _parse_roi(level_roi)
    print(f"레벨 크롭 좌표: x={x}, y={y}, w={w}, h={h}")

    fh, fw = frame.shape[:2]
    x2 = min(x + w, fw)
    y2 = min(y + h, fh)

    if x < 0 or y < 0 or x >= fw or y >= fh:
        print(f"ERROR: 레벨 ROI 가 프레임 밖입니다! frame=({fw},{fh})")
    else:
        crop_lv = frame[y:y2, x:x2]
        print(f"crop shape: {crop_lv.shape}")

        # 원본 크롭 저장
        cv2.imwrite('debug_level_crop.png', crop_lv)
        print("→ debug_level_crop.png 저장됨 (실제로 뭘 보고 있는지 확인)")

        # 전처리 (레퍼런스 level_reader.py _preprocess 와 동일)
        h2, w2 = crop_lv.shape[:2]
        enlarged = cv2.resize(crop_lv, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LINEAR)
        gray     = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite('debug_level_processed.png', binary)
        print("→ debug_level_processed.png 저장됨 (전처리: 2배확대+CLAHE+OTSU)")

        # ── easyocr ──────────────────────────────────────────────────────
        try:
            import easyocr
            print("\neasyocr 초기화 중...")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("easyocr 초기화 완료\n")

            # 전처리 이미지 OCR (detail=1 — 레퍼런스 level_reader.py 방식)
            print("[easyocr] 전처리 이미지 OCR (detail=1):")
            results = reader.readtext(binary, detail=1, paragraph=False)
            if results:
                for bbox, text, conf in results:
                    mark = "✅" if conf >= 0.1 else "❌(conf<0.1)"
                    print(f"  텍스트='{text}'  신뢰도={conf:.2f}  {mark}")
            else:
                print("  (결과 없음)")

            # 원본 crop OCR (참고용)
            print("\n[easyocr] 원본 crop OCR (detail=1, 참고용):")
            results2 = reader.readtext(crop_lv, detail=1, paragraph=False)
            if results2:
                for bbox, text, conf in results2:
                    mark = "✅" if conf >= 0.1 else "❌(conf<0.1)"
                    print(f"  텍스트='{text}'  신뢰도={conf:.2f}  {mark}")
            else:
                print("  (결과 없음)")

        except ImportError as e:
            print(f"[easyocr] 미설치: {e}")
        except Exception as e:
            print(f"[easyocr] 오류: {e}")

        # ── tesseract (참고용) ────────────────────────────────────────────
        try:
            import pytesseract
            tess_cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:/LEVHPlevhp"
            lv_text  = pytesseract.image_to_string(binary, config=tess_cfg).strip()
            print(f"\n[tesseract] 레벨 OCR: \"{lv_text}\"")
        except ImportError:
            print("\n[tesseract] 미설치 (선택사항)")
        except Exception as e:
            print(f"\n[tesseract] 오류: {e}")

else:
    print("\n⚠️  level_roi 가 null — 레벨 OCR 진단 건너뜀")

print("\n" + "="*60)
print("  진단 완료")
print("="*60)
print("저장된 파일:")
print("  debug_level_crop.png      — 레벨 원본 크롭")
print("  debug_level_processed.png — 레벨 전처리 (2배확대+CLAHE+OTSU)")
print("  debug_hp_crop.png         — HP 원본 크롭")
print("  debug_hp_mask.png         — HP 빨간색 마스크")
print("  debug_hp_overlay.png      — HP 원본+마스크 overlay")
print("\n→ 이미지 파일을 채팅창에 드래그앤드롭으로 공유하면 진단 가능합니다.")
