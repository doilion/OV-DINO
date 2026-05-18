import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model

model_root = os.getenv("MODEL_ROOT", "./inits")
# OV-DINO pretrained bundle contains Swin-T backbone + DINO transformer +
# class/bbox heads + BERT — strictly stronger init than standalone Swin alone.
init_checkpoint = osp.join(
    model_root, "ovdino", "ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth"
)

# get default config
dataloader = get_config("common/data/tct_ngc_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# LR schedule: 24 epochs, decay at epoch 16 and 22, 1000-step warmup
# 116524 samples / 8 batch_size = 14566 iters/epoch, 24 epochs = 349584 iters
from configs.common.coco_schedule import multi_steps_scheduler

lr_multiplier = multi_steps_scheduler(24, [16, 22], 1000, 116524, 8)

# modify training config
train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_base30_640_24ep"

# max training iterations: ceil(116524 / 8) * 24 = 14566 * 24 = 349584
train.max_iter = 349600
train.eval_period = 14566
train.log_period = 50
train.checkpointer.period = 14566

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 30
model.test_num_classes = 30

# amp
train.amp.enabled = True

# modify optimizer config
optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: (
    0.1 if "language_backbone" in module_name else 1.0
)

# modify dataloader config
dataloader.train.num_workers = 4

# total batch size: 8 GPUs x 1 per GPU = 8
# (640+1024 still leaves bs=2/GPU peaking at ~10.9GB on 11GB 2080Ti — unsafe)
dataloader.train.total_batch_size = 8

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
