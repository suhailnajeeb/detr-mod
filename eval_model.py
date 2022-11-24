from engine import evaluate
from models import build_model
from datasets import build_dataset

import util.misc as utils

from torch.utils.data import DataLoader
from datasets import get_coco_api_from_dataset

import argparse
import torch

from test_utils import load_model_all_from_ckp
from datasets.coco_eval import CocoEvaluator

from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None,
                        help="Number of classes in dataset+1")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/Users/muhammadsuha/projects/detr-mod/outputs/JICCCNU_01/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()


    model_path = args.output_dir + 'checkpoint.pth'

    model, criterion, postprocessors = load_model_all_from_ckp(model_path)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    device = torch.device(args.device)

    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, \
        data_loader_val, base_ds, device, args.output_dir)

    # # EVALUATION STARTS HERE

    # model.eval()
    # criterion.eval()

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    # #breakpoint()

    # coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # i = 0

    # for samples, targets in tqdm(data_loader_val):
    #     # Transfer to device
    #     samples = samples.to(device)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #     # Get predictions
    #     outputs = model(samples)
    #     loss_dict = criterion(outputs, targets)
    #     weight_dict = criterion.weight_dict         # Why do we need this again?


    #     orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    #     results = postprocessors['bbox'](outputs, orig_target_sizes)

    #     res = {target['image_id'].item(): output for target, output in zip(targets, results)}

    #     if coco_evaluator is not None:
    #         coco_evaluator.update(res)
        
    #     i+= 1

    #     if(i == 10):
    #         breakpoint()

    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    
    # breakpoint()