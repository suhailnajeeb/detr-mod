import os
import json
import random
from PIL import Image
from test_utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision

# Next, define a function for NMS
def nms(boxes, scores, iou_threshold):
    # Keep only the scores for the maximum class for each prediction
    scores, _ = scores.max(dim=-1)
    # Perform NMS using the torchvision.ops.nms function
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return keep

CLASSES = ["D00", "D10", "D20", "D40"]

data_root = '/Users/muhammadsuha/Datasets/RDD-2022/holdout/coco_holdout'
set_name = 'test2017'
model_dir = 'outputs/combined_00/'

annotations_path = os.path.join(
    data_root, 'annotations/instances_{}.json'.format(set_name))

thresh = 0.5

#file_name = 'Japan_011728.jpg'
file_name = None

# load annotations

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

images = annotations['images']

if not file_name:
    image = random.choice(images)
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

model_path = model_dir + 'checkpoint.pth'
model, criterion, postprocessors = load_model_all_from_ckp(model_path)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with thresh+ confidence
proas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > thresh

# convert all boxes to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, :], im.size)

# Get the final set of prediction boxes and scores by indexing into bboxes_scaled and probas using the keep tensor
prediction_boxes = bboxes_scaled[keep]
prediction_scores = probas[keep]

# Perform NMS with an IoU threshold of 0.5
keep_nms = nms(prediction_boxes, prediction_scores, iou_threshold=0.5)

# Keep only the top-scoring predictions
prediction_boxes = prediction_boxes[keep_nms]
prediction_scores = prediction_scores[keep_nms]

# convert all boxes to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, :], im.size)

colors = COLORS * 100

#ax = plt.gca()

fig, ax = plt.subplots(1, 3, figsize=(30, 10))

ax[0].imshow(im)
ax[0].set_title('Ground Truth')

for a in annotations:
    x, y, w, h = a['bbox']
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

ax[2].imshow(im)
ax[2].set_title('Predictions with NMS')

# Plot the final set of prediction boxes and scores
for i in range(len(prediction_boxes)):
    bbox = prediction_boxes[i]
    prob = prediction_scores[i]
    class_id = prob.argmax()
    class_name = CLASSES[class_id]
    prob = prob[class_id]*100
    color = colors[class_id]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
    ax[2].add_patch(rect)
    ax[2].text(bbox[0], bbox[1], '{}: {:.0f} %'.format(class_name, prob), bbox={'facecolor': color, 'alpha': 0.5})
ax[2].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
plt.axis('off')

plt.show()