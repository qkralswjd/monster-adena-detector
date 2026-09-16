import json
cfg = json.load(open('config.json', encoding='utf-8'))
cfg['level_detector']['level_roi'] = {'x': 238, 'y': 863, 'w': 109, 'h': 30}
cfg['level_detector']['hp_roi']    = {'x': 548, 'y': 842, 'w': 335, 'h': 53}
json.dump(cfg, open('config.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('저장완료')
print('레벨ROI:', cfg['level_detector']['level_roi'])
print('HP ROI:', cfg['level_detector']['hp_roi'])
