import json

pred_path = '/data/gpfs/projects/punim1800/detr-mod/outputs/holdout_00/predictions.json'

with open(pred_path) as f:
    pred = json.load(f)

