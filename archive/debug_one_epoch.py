from models import build_model
import util.misc as utils
import torch

class ArgsClass:
    def __init__(self):
        self.dataset_file = 'coco'
        self.device = 'cpu'
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


args = ArgsClass()

# inputs to the function

data_loader = ''        # We will replace the dataloader with our own thingy
optimizer = ''
device = ''


dataset_train = build_dataset(image_set='train', args=args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)

batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)



model, criterion, postprocessors = build_model(args)





# set to train mode

model.train()
criterion.train()

for sample, targets in data_loader:
    samples  = samples.to(device)
    targets = targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    optimizer.zero_grad()
    losses.backward()

    optimizer.step()


