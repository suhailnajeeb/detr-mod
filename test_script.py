# Library imports

import os
import json
import random
from PIL import Image
from test_utils import *
import matplotlib.pyplot as plt

# Steps for testing script 

data_root = '/Users/muhammadsuha/Datasets/RDD-2022/coco_JP_IN_CZ_CM_CD_NW_US/'
set_name = 'val2017'
model_dir = 'outputs/JICCCNU_01/'

annotations_path = os.path.join(
    data_root, 'annotations/instances_{}.json'.format(set_name))

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

images = annotations['images']
image = random.choice(images)
file_name = image['file_name']
image_id = image['id']

# get annotations for the image_id
annotations = [a for a in annotations['annotations'] if a['image_id'] == image_id]

image_path = os.path.join(data_root, set_name, file_name)

im = Image.open(image_path)

# Load Model

model_path = model_dir + 'checkpoint.pth'
model = load_model_from_ckp(model_path)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)


outputs = model(img)

probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

fig = plot_gt_preds(im, annotations, probas, bboxes_scaled)

# Evaluate

plt.show()