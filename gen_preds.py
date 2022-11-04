import os
import json
import random
from PIL import Image
from test_utils import *
from models import build_model

CLASSES = ["D00", "D10", "D20", "D40"]

#random.seed()

data_root = '/Users/muhammadsuha/Datasets/RDD-2022/coco_JP_IN_CZ_CM_CD_NW_US/'
set_name = 'val2017'
model_dir = 'outputs/JICCCNU_01/'

annotations_path = os.path.join(
    data_root, 'annotations/instances_{}.json'.format(set_name))

# load annotations

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

images = annotations['images']
image = random.choice(images)
file_name = image['file_name']
image_id = image['id']

# get annotations for the image_id
annotations = [a for a in annotations['annotations'] if a['image_id'] == image_id]

# load image and plot annotations in image
image_path = os.path.join(data_root, set_name, file_name)

im = Image.open(image_path)

# load model

#model_path = 'outputs/checkpoint.pth'
model_path = model_dir + 'checkpoint.pth'
model = load_model_from_ckp(model_path)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img )

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# plot predictions

fig = plot_gt_preds(im, annotations, probas, bboxes_scaled)

# save figure

fig.savefig(model_dir + '{}.png'.format(file_name))

fig.show()