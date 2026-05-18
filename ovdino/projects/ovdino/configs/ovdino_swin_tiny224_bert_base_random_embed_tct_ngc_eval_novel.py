"""Novel-class eval config for the random-embedding ablation.

The checkpoint was trained with 20 base-class embeddings. We evaluate on the
11 novel classes by loading an extended checkpoint (model_0078299_extended31.pth)
whose embedding table has been padded with 11 untrained random rows. Novel AP
is expected to be near zero — this is intentional, serving as a baseline that
demonstrates random embeddings cannot generalise to unseen classes.
"""
import os

from detrex.config import get_config
from detrex.data.datasets.register_tct_ngc_ovd import (
    TCT_NGC_BASE_CATEGORIES,
    TCT_NGC_NOVEL_CATEGORIES,
)
from detrex.data.datasets.utils import clean_words_or_phrase

from .models.ovdino_swin_tiny224_bert_base import model

dataloader = get_config("common/data/tct_ngc_novel_ovd.py").dataloader
train = get_config("common/train.py").train

train.device = "cpu"
model.device = train.device
model.num_classes = 20
model.test_num_classes = 11

# Drop BERT; use the extended 31-entry random embedding table.
model.language_backbone = None
model.use_random_text_embedding = True
model.random_vocab_names = (
    [clean_words_or_phrase(c["name"]) for c in TCT_NGC_BASE_CATEGORIES]
    + [clean_words_or_phrase(c["name"]) for c in TCT_NGC_NOVEL_CATEGORIES]
)
model.random_embed_init_std = 0.02
model.freeze_random_embedding = True  # eval only — no grad needed

dataloader.evaluator.output_dir = train.output_dir
