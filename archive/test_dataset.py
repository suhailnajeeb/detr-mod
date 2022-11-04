import torch
import tqdm
from torch.utils.data import DataLoader
from datasets import build_dataset
import util.misc as utils
import util.box_ops as box_ops

class ArgsClass:
    def __init__(self, coco_path):
        self.dataset_file = 'coco'
        self.batch_size = 1
        self.num_workers = 0
        self.pin_memory = False
        self.shuffle = False
        self.coco_path = coco_path
        self.masks = False

coco_path = "/Users/muhammadsuha/datasets/RDD-2022/coco"

args = ArgsClass(coco_path)

dataset_train = build_dataset(image_set='train', args=args)

#sampler_train = torch.utils.data.RandomSampler(dataset_train)

#batch_sampler_train = torch.utils.data.BatchSampler(
#        sampler_train, args.batch_size, drop_last=True)

#data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
#                                collate_fn=utils.collate_fn, num_workers=args.num_workers)

n_dataset = len(dataset_train)

dataset_iter = iter(dataset_train)

for i in tqdm.tqdm(range(n_dataset)):
    image, target = next(dataset_iter)
    try:
        xyxy = box_ops.box_cxcywh_to_xyxy(target['boxes'])
    except:
        print("Error with data index: ", i)