"""Reference eval config: original OV-DINO BERTEncoder × long PSC prompts.

A/B counterpart to the stock custom-dataset eval config. **Nothing about
the model changes** — same SwinT + BERT-base text encoder, same
checkpoint, same dataloader. The only difference: at DatasetCatalog
load time, every record's ``category_names`` is rewritten from the bare
class name (e.g. ``"respiratory tract-Neutrophil"``) to a long
descriptive PSC prompt (e.g. ``"Respiratory tract cytology - Neutrophil"``).

This is the ablation config for "does feeding richer language into
OV-DINO's stock BERT improve detection on a clinical dataset?"

Env vars:
  * ``OVDINO_PSC_NAME2PROMPT``   absolute path to the prompt JSON
                                 (default: data/texts/tct_ngc_class_prompts_base30.json
                                  resolved relative to repo root).
  * ``OVDINO_PSC_STRICT``        "0" to allow classes missing from the
                                 JSON (keeps bare names for the misses).
                                 Default strict — every class must map.
"""

import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model

model_root = os.getenv("MODEL_ROOT", "./inits")
init_checkpoint = osp.join(model_root, "./swin", "swin_tiny_patch4_window7_224.pth")

dataloader = get_config("common/data/custom_ovd.py").dataloader
train = get_config("common/train.py").train

train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_bert_base_eval_psc_prompts"
train.device = "cuda"
model.device = train.device
dataloader.evaluator.output_dir = train.output_dir

# Inference template must stay "identity" — the JSON keys are the literal
# prompts we want BERT to encode, no "a photo of a {}" wrapping on top.
model.inference_template = "identity"

# ---- PSC prompt substitution at registration time ---------------------
from projects.ovdino.data.psc_prompt_register import attach_psc_prompts_to_dataset
from omegaconf import ListConfig, OmegaConf

# Default points at the in-repo base30 file; override via env var when
# evaluating against the 32-class test set or a different dataset.
name_to_prompt_path = os.getenv(
    "OVDINO_PSC_NAME2PROMPT", "data/texts/tct_ngc_class_prompts_base30.json"
)
strict = os.getenv("OVDINO_PSC_STRICT", "1") != "0"

names = dataloader.test.dataset.names
if isinstance(names, ListConfig):
    names = OmegaConf.to_container(names, resolve=True)
if isinstance(names, str):
    names = [names]
for n in names:
    attach_psc_prompts_to_dataset(n, name_to_prompt_path, strict=strict)
