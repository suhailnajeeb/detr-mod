This is just a place to keep track of the models that we have been training 
so far. 

JICCNU_01 : Fine-tuning on pretrained DETR with all data.
JICCNU_00: Training DETR from scratch using default schedules with all data. 


JICCNU_256: Pre-trained DETR with Fine-tuning on 256x256 data.

combined_00: Training with default DETR high resolutions
                Learning rate: 1e-5         LR backbone: 1e-6
                Let's train a model with a higher Learning rate? 

combined_01: Training with default DETR mixed resolutions
                Learning rate: 1e-4         LR backbone: 1e-5
                (This model will be up for training next)
                Resize: Capped to 640 min, max - 1200

combined_02: Training with previous DETR settings
                Learning rate: 1e-5         LR backbone: 1e-6
                Resize: hard-resize to 600x600

combined_03: Training with previous DETR settings
                Learning rate: 1e-5         LR backbone: 1e-6
                Resize: hard-resize to 800x800

combined_04: Training with default DETR settings
                Learning rate: 1e-5         LR backbone: 1e-6
                Resize: coco_transforms with scales up to 600
                and max-size: 600

combined_05: Training with default DETR settings
                Learning rate: 1e-5         LR backbone: 1e-6
                Resize: coco_transforms with scales up to 800
                and max-size: 800

holdout_00:
- job_id: 
- default DETR settings
- LR: 1e-5      LR backbone: 1e-6
- Rescale/Random Crop: Yes  up to 800
- max-size: 1200

holdout_01:
- job_id: 42559591
- command: sbatch train_holdout.slurm
- default DETR settings
- LR: 1e-5      LR backbone: 1e-6
- Rescale/Random Crop: Yes  up to 800
- max-size: 1333
