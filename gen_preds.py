import os
import json
import random
from PIL import Image
from test_utils import *
from models import build_model
import matplotlib.pyplot as plt

CLASSES = ["D00", "D10", "D20", "D40"]

#random.seed()

data_root = '/Users/muhammadsuha/Datasets/RDD-2022/coco_JP_IN_CZ_CM_CD_NW_US/'
set_name = 'val2017'
model_dir = 'outputs/JICCCNU_01/'

annotations_path = os.path.join(
    data_root, 'annotations/instances_{}.json'.format(set_name))

thresh = 0.5
sample = None

file_name_list = [
    'United_States_003152.jpg',
    'United_States_002715.jpg',
    'Japan_007891.jpg',
    'Japan_002552.jpg',
    'Japan_002219.jpg',
    'United_States_004726.jpg',
    'Japan_004583.jpg',
    'United_States_003272.jpg',
    'United_States_002142.jpg',
    'Japan_005071.jpg',
    'Japan_010756.jpg',
    'United_States_004325.jpg'
    'Japan_012478.jpg',
    'Japan_007148.jpg',
    'United_States_004012.jpg'
    'Japan_013071.jpg'
    'United_States_003545.jpg'
]

#sample = random.choice(list(range(len(file_name_list))))

if sample is not None:
    file_name = file_name_list[sample]
else:
    file_name = None

# load annotations

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

images = annotations['images']
image = random.choice(images)
if not file_name:
    file_name = image['file_name']
    image_id = image['id']
    print('file_name: ', file_name)
else:
    image_id = get_image_id(file_name)

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
outputs = model(img)

# keep only predictions with thresh+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > thresh

# convert boxes from [0; 1] to image scales
#bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# convert all boxes to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, :], im.size)

# plot predictions

#fig = plot_gt_preds(im, annotations, probas, bboxes_scaled)

# save figure

#fig.savefig(model_dir + '{}.png'.format(file_name))

# Plot the predicted bounding boxes along with the probability of the predicted class

import matplotlib.pyplot as plt
import matplotlib.patches as patches

colors = COLORS * 100

#ax = plt.gca()

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(im)
ax[0].set_title('Ground Truth')

for a in annotations:
    x, y, w, h = a['bbox']
    #print(a['category_id'])
    #class_name = CLASS_NAMES[a['category_id']]
    class_code = CLASSES[a['category_id']]
    color = colors[a['category_id']]
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax[0].add_patch(rect)
    ax[0].text(x, y, class_code, bbox={'facecolor': color, 'alpha': 0.5})
ax[0].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
plt.axis('off')

ax[1].imshow(im)
ax[1].set_title('Predictions')

for i in range(len(keep)):
    if keep[i]:
        bbox = bboxes_scaled[i]
        prob = probas[i]
        class_id = prob.argmax()
        class_name = CLASSES[class_id]
        prob = prob[class_id]*100
        color = colors[class_id]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
        ax[1].add_patch(rect)
        ax[1].text(bbox[0], bbox[1], '{}: {:.0f} %'.format(class_name, prob), color=color, bbox=dict(facecolor='white', alpha=0.5))

# plot legend including CLASS_NAMES

ax[1].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
plt.axis('off')

plt.show()