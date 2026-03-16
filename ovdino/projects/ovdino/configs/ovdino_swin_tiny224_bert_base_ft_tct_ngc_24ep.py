import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model

model_root = os.getenv("MODEL_ROOT", "./inits")
init_checkpoint = osp.join(model_root, "./swin", "swin_tiny_patch4_window7_224.pth")

# get default config
dataloader = get_config("common/data/tct_ngc_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# LR schedule: 24 epochs, decay at epoch 16 and 22, 1000-step warmup
# 69590 samples / 8 batch_size = 8699 iters/epoch, 24 epochs = 208776 iters
from configs.common.coco_schedule import multi_steps_scheduler

lr_multiplier = multi_steps_scheduler(24, [16, 22], 1000, 69590, 8)

# modify training config
train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_bert_base_24ep_ft_tct_ngc"

# max training iterations: 69590 / 8 * 24 = 208770 -> 208800
train.max_iter = 208800
train.eval_period = 8700
train.log_period = 50
train.checkpointer.period = 8700

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 20

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
# (conservative for 2080Ti 11GB with 4112x3008 images resized to ~800x1093)
dataloader.train.total_batch_size = 8

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
