import torch
from models import build_model

class ArgsClass:
    def __init__(self):
        self.dataset_file = 'coco'
        self.device = 'cuda'
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
        self.num_queries = 100
        self.aux_loss = ''
        self.num_classes = 5
        self.frozen_weights = None
        self.giou_loss_coef = 2
        self.masks = False
        self.aux_loss = False
        # We will need additional terms if the above is true
        self.distributed = None
        self.num_workers = 1
        self.pre_norm = False
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1


args = ArgsClass()

device = 'cuda'

model, criterion, postprocessors = build_model(args)
model.to(device)

model_path = 'outputs/holdout_00/checkpoint.pth'
state_dict = torch.load(model_path)['model']
model.load_state_dict(state_dict)

model.eval()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f'Total Parameters: {total_params}')
print(f'Trainable Parameters: {trainable_params}')