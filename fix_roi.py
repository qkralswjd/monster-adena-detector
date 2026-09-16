import json
cfg = json.load(open('config.json', encoding='utf-8'))
cfg['level_detector']['level_roi'] = {'x': 242, 'y': 874, 'w': 66, 'h': 23}
cfg['level_detector']['hp_roi']    = {'x': 714, 'y': 859, 'w': 116, 'h': 21}
json.dump(cfg, open('config.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('저장완료')
print('레벨ROI:', cfg['level_detector']['level_roi'])
print('HP ROI:', cfg['level_detector']['hp_roi'])
