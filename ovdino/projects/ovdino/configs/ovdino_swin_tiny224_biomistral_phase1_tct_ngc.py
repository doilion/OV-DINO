"""Phase 1: Freeze visual, train Adapter MLP + ClassEmbed + BBoxEmbed.

Loads pre-aligned adapter weights from Phase 0. Visual backbone, neck,
transformer, position embedding, and label encoder are frozen.

8 epochs, with correspondence loss enabled.
"""
import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_biomistral import model

model_root = os.getenv("MODEL_ROOT", "./inits")
# Load OV-DINO pre-trained checkpoint (ClassEmbed weights preserved)
init_checkpoint = osp.join(model_root, "ovdino", "ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth")

# get default config
dataloader = get_config("common/data/tct_ngc_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# LR schedule: 8 epochs, decay at epoch 6, 500-step warmup
# 69590 samples / 8 batch_size = 8699 iters/epoch, 8 epochs = 69592 iters
from configs.common.coco_schedule import multi_steps_scheduler

lr_multiplier = multi_steps_scheduler(8, [6], 500, 69590, 8)

# training config
train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_biomistral_phase1_tct_ngc"
train.max_iter = 69600  # ~8 epochs
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

# Phase 1: freeze visual components
model.freeze_visual = True

# Load pre-aligned adapter weights from Phase 0
model.adapter_init_checkpoint = "embeddings/adapter_prealigned.pth"

# AMP
train.amp.enabled = True

# DDP: need find_unused_parameters since visual backbone is frozen
train.ddp.find_unused_parameters = True

# optimizer: adapter_mlp gets higher LR (randomly initialized but pre-aligned)
# class_embed and bbox_embed at standard LR
# frozen visual params excluded automatically (requires_grad=False)
optimizer.lr = 1e-4  # base LR for adapter_mlp
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: (
    0.1
    if "class_embed" in module_name or "bbox_embed" in module_name
    else 1.0
)

# dataloader
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 8
dataloader.evaluator.output_dir = train.output_dir
