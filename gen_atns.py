# Library Imports
import matplotlib.pyplot as plt
from PIL import Image
import requests
from test_utils import *
import random
import json
import os

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

CLASSES = ["D00", "D10", "D20", "D40"]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

random.seed()

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

#plot_results(im, probas[keep], bboxes_scaled)

fig = plot_gt_preds(im, annotations, probas, bboxes_scaled)

#plt.show()

# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []

hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]

# propagate through the model
outputs = model(img)

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0]
dec_attn_weights = dec_attn_weights[0]

# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]
colors = COLORS * 100

if len(bboxes_scaled) > 1:
    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    fig.tight_layout()
elif len(bboxes_scaled) == 1:
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(7, 7))
    for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
        ax = axs[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = axs[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
else:
    print('No object detected')

# set title of the figure
fig.suptitle('Decoder Attention Weights', fontsize=16)

#plt.show()

# output of the CNN
f_map = conv_features['0']
print("Encoder attention:      ", enc_attn_weights[0].shape)
print("Feature map:            ", f_map.tensors.shape)


# get the HxW shape of the feature maps of the CNN
shape = f_map.tensors.shape[-2:]
# and reshape the self-attention to a more interpretable shape
sattn = enc_attn_weights[0].reshape(shape + shape)
print("Reshaped self-attention:", sattn.shape)


# downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
fact = 32

# get the center of the bounding boxes

# for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
#     # get the center of the bounding box
#     center = torch.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2])
#     # and the class of the bounding box
#     cls = probas[idx].argmax()
#     # and the attention weights for the bounding box
#     attn = sattn[int(center[1] / fact), int(center[0] / fact)]
#     # plot the attention weights
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(attn)
#     ax.set_title(f'Attention weights for class {CLASSES[cls]}')
#     ax.axis('off')
#     fig.tight_layout()

centers = []

for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
    # get the center of the bounding box
    center = [int(((ymin + ymax) // 2).item()), int(((xmin + xmax) // 2).item())]
    centers.append(center)
    if len(centers) == 4:
        break
        print('There are more than 4 objects in the image')

#print(centers)

# let's select 4 reference points for visualization
points = [(200, 200),(200, 400), (400, 200), (400,400)]# (280, 400), (200, 600), (440, 800),]

idxs = centers

while len(idxs) < 4:
    idxs.append(random.choice(points))

# here we create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# and we add one plot per reference point
gs = fig.add_gridspec(2, 4)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[0, -1]),
    fig.add_subplot(gs[1, -1]),
]

# for each one of the reference points, let's plot the self-attention
# for that point
for idx_o, ax in zip(idxs, axs):
    idx = (idx_o[0] // fact, idx_o[1] // fact)
    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(im)
for (y, x) in idxs:
    scale = im.height / img.shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')

plt.show()