"""Phase 2: Unfreeze all, joint fine-tuning.

Loads Phase 1 checkpoint. All components are trainable with differentiated
learning rates. Visual backbone gets 0.1x LR to prevent catastrophic forgetting.

16 epochs, with correspondence loss enabled.
"""
import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_biomistral import model

# Load Phase 1 checkpoint
init_checkpoint = "./wkdrs/ovdino_biomistral_phase1_tct_ngc/model_final.pth"

# get default config
dataloader = get_config("common/data/tct_ngc_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# LR schedule: 16 epochs, decay at epoch 11 and 14, 500-step warmup
# 69590 samples / 8 batch_size = 8699 iters/epoch, 16 epochs = 139184 iters
from configs.common.coco_schedule import multi_steps_scheduler

lr_multiplier = multi_steps_scheduler(16, [11, 14], 500, 69590, 8)

# training config
train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_biomistral_phase2_tct_ngc"
train.max_iter = 139200  # ~16 epochs
train.eval_period = 8700
train.log_period = 50
train.checkpointer.period = 8700

# gradient clipping
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# device
train.device = "cuda"
model.device = train.device
model.num_classes = 20

# Phase 2: NO freezing (all components trainable)
model.freeze_visual = False

# AMP
train.amp.enabled = True

# optimizer: differentiated learning rates
# backbone gets 0.1x, adapter_mlp at base, rest at base
optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: (
    0.1
    if "backbone" in module_name and "language_backbone" not in module_name
    else 1.0
)

# dataloader
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 8
dataloader.evaluator.output_dir = train.output_dir
