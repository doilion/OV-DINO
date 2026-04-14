from detrex.config import get_config

from .models.ovdino_swin_tiny224_biomistral import model

# get default config — novel evaluation dataset
dataloader = get_config("common/data/tct_ngc_novel_ovd.py").dataloader
train = get_config("common/train.py").train

# model config (must match training checkpoint)
train.device = "cuda"
model.device = train.device
model.num_classes = 20

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
