from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model

# get default config
dataloader = get_config("common/data/tct_ngc_novel_ovd.py").dataloader
train = get_config("common/train.py").train

# model config (must match training checkpoint)
train.device = "cuda"
model.device = train.device
# Trained on 30 base classes; eval on 5 novel classes (zero-shot).
model.num_classes = 30
model.test_num_classes = 5

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
