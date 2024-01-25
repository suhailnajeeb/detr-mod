To train baseline DETR on a single node with 8 gpus for 300 epochs run:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco

python main.py --coco_path /Users/muhammadsuha/04-Archive/Datasets/VisDrone2019/VisDrone2019-COCO --device cpu