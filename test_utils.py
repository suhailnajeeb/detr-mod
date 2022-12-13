import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

from models import build_model

# COCO classes
CLASSES_COCO = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

CLASSES = ["D00", "D10", "D20", "D40"]

CLASS_NAMES = {
    0: "D00: Longitudinal crack",
    1: "D10: Transverse crack",
    2: "D20: Alligator crack",
    3: "D40: Pothole"
}

country_dict = {
    'China_Drone': '1',
    'China_MotorBike': '2',
    'Czech': '3',
    'India': '4',
    'Japan': '5',
    'Norway': '6',
    'United_States': '7'
}

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Standard PyTorch Transforms
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(600),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_image_annotation(image, annotations):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    for a in annotations:
        x, y, w, h = a['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        class_name = CLASSES[a['category_id'] - 1]
        ax.text(x, y, class_name, bbox={'facecolor': 'red', 'alpha': 0.5})
    plt.axis('off')
    plt.show()

class ArgsModel:
    def __init__(self):
        self.dataset_file = 'coco'
        self.num_classes = 5
        self.device = 'cuda'
        self.num_queries = 100
        self.aux_loss = True
        self.masks = False
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1
        self.hidden_dim = 256
        self.position_embedding = 'sine'
        self.lr_backbone = 1e-5
        self.backbone = 'resnet50'
        self.dilation = False
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2

def load_model_from_ckp(ckp_path):
    model, criterion, postprocessors = build_model(ArgsModel())
    model.load_state_dict(torch.load(ckp_path, map_location='cpu')['model'])
    model.eval();
    return model

def load_model_all_from_ckp(ckp_path):
    model, criterion, postprocessors = build_model(ArgsModel())
    model.load_state_dict(torch.load(ckp_path, map_location = 'cpu')['model'])
    model.eval();
    return model, criterion, postprocessors

from matplotlib import pyplot as plt
import matplotlib.patches as patches

colors = COLORS * 100

def plot_gt_preds(im, annotations, probas, bboxes_scaled):
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

    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
        cl = p.argmax()
        print('class', cl)
        print(p)
        color = colors[cl]
        ax[1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=color, linewidth=2))
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax[1].text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    ax[1].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])
    plt.axis('off')

    return fig

# def plot_gt_preds2(im, annotations, probas, bboxes_scaled):
#     fig, ax = plt.subplots(1, 3, figsize=(30, 10))

#     ax[0].imshow(im)
#     ax[0].set_title('Ground Truth')

#     for a in annotations:
#         x, y, w, h = a['bbox']
#         #print(a['category_id'])
#         #class_name = CLASS_NAMES[a['category_id']]
#         class_code = CLASSES[a['category_id']]
#         color = colors[a['category_id']]
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
#         ax[0].add_patch(rect)
#         ax[0].text(x, y, class_code, bbox={'facecolor': color, 'alpha': 0.5})
#     ax[0].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
#     plt.axis('off')

#     ax[1].imshow(im)
#     ax[1].set_title('Predictions')

#     for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
#         cl = p.argmax()
#         color = colors[cl]
#         ax[1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                     fill=False, color=color, linewidth=2))
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax[1].text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     ax[1].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])
#     plt.axis('off')



    return fig

# Function to get image_id from file_name

def get_country(file_name):
    file_name = file_name.split('.')[0]
    return file_name[:-7]

#def get_image_id(file_name):
#    file_name = file_name.split('.')[0]
#    return file_name[-6:]

def get_image_id(file_name):
    file_name = file_name.split('.')[0]
    country = file_name[:-7]
    image_id = file_name[-6:]
    country_id = country_dict[country]
    return int(country_id + image_id)