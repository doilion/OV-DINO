"""Random-embedding ablation on TCT_NGC base (30 classes).

Identical to ovdino_swin_tiny224_bert_base_ft_tct_ngc_24ep.py except the
BERT text encoder is replaced by a per-class learnable embedding table
(nn.Embedding(30, 768)) initialised from N(0, 0.02^2). No text signal.

random_vocab_names auto-tracks TCT_NGC_BASE_CATEGORIES — after the prompt-
mapping change in register_tct_ngc_ovd.py, the vocab keys are now the
clinical prompt forms (cleaned). That's fine for random-embed because the
table only needs a stable name->row mapping; the actual text content is
ignored downstream.
"""
import os
import os.path as osp

from detrex.config import get_config
from detrex.data.datasets.register_tct_ngc_ovd import TCT_NGC_BASE_CATEGORIES
from detrex.data.datasets.utils import clean_words_or_phrase

from .models.ovdino_swin_tiny224_bert_base import model

model_root = os.getenv("MODEL_ROOT", "./inits")
# Use full OV-DINO pretrained (Swin + transformer + heads); BERT weights in
# the checkpoint are ignored since language_backbone=None.
init_checkpoint = osp.join(
    model_root, "ovdino", "ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth"
)

dataloader = get_config("common/data/tct_ngc_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# LR schedule: match BERT baseline (24 epochs, decay at 16, 22; 1000-step warmup)
# 116524 samples / 8 batch_size = 14566 iters/epoch, 24 epochs = 349584 iters
from configs.common.coco_schedule import multi_steps_scheduler

lr_multiplier = multi_steps_scheduler(24, [16, 22], 1000, 116524, 8)

train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_bert_base_random_embed_tct_ngc_base30_640_24ep"
train.max_iter = 349600
train.eval_period = 14566
train.log_period = 50
train.checkpointer.period = 14566

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device
model.num_classes = 30
model.test_num_classes = 30

train.amp.enabled = True

# Drop the BERT text encoder: saves ~110M params, ~400MB GPU, and avoids
# wasted HF download at startup. The new random-embedding path is enabled
# by the three flags below.
model.language_backbone = None
model.use_random_text_embedding = True
model.random_vocab_names = [
    clean_words_or_phrase(c["name"]) for c in TCT_NGC_BASE_CATEGORIES
]
model.random_embed_init_std = 0.02
model.freeze_random_embedding = False

optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
# BERT baseline applied 0.1x LR to language_backbone; with no text encoder the
# predicate never fires. The learnable table gets the base LR. Bump it if the
# class logits look under-trained in sanity check.
optimizer.params.lr_factor_func = lambda module_name: (
    0.1 if "language_backbone" in module_name else 1.0
)

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 8
dataloader.evaluator.output_dir = train.output_dir
