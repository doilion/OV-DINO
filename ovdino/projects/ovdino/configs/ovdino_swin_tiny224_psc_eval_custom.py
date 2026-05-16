"""Reference eval config: OV-DINO with pre-computed PSC text embeddings.

Swaps ``model.language_backbone`` from the in-loop ``BERTEncoder`` to a
``PseudoLanguageBackbone`` that serves cached embeddings. Use this when:

  * the class set is fixed (custom medical dataset),
  * you have rich descriptive prompts (TBS-style full names),
  * **or** you want to evaluate with BiomedCLIP / PubMedBERT-encoded text
    without dragging those packages into the training runtime.

Env vars:
  * ``OVDINO_PSC_CACHE``         absolute path to the cache .pth file
  * ``OVDINO_PSC_NAME2PROMPT``   (optional) JSON ``{class_name: prompt}``
                                 — set this if the cache is keyed by long
                                 descriptive prompts but the dataset's
                                 ``category_names`` are bare class names.
  * ``OVDINO_PSC_DIM``           expected D of the cached vectors
                                 (default 768; set 512 for BiomedCLIP).
  * ``OVDINO_PSC_NORMALIZE``     "1" to L2-normalize at lookup
                                 (default off — assume cache already
                                 normalized).
"""

import os
import os.path as osp

from detectron2.config import LazyCall as L
from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model
from projects.ovdino.modeling.pseudo_language_backbone import (
    PseudoLanguageBackbone,
)

model_root = os.getenv("MODEL_ROOT", "./inits")
init_checkpoint = osp.join(model_root, "./swin", "swin_tiny_patch4_window7_224.pth")

dataloader = get_config("common/data/custom_ovd.py").dataloader
train = get_config("common/train.py").train

train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_psc_eval_custom"
train.device = "cuda"
model.device = train.device
dataloader.evaluator.output_dir = train.output_dir

# ---- PSC language backbone wiring --------------------------------------
psc_cache = os.getenv("OVDINO_PSC_CACHE")
if not psc_cache:
    raise SystemExit(
        "OVDINO_PSC_CACHE is required by ovdino_swin_tiny224_psc_eval_custom.py "
        "— build a cache via projects/ovdino/data/build_text_embeddings.py "
        "and export its path before running scripts/eval.sh."
    )

psc_dim = int(os.getenv("OVDINO_PSC_DIM", "768"))
psc_normalize = os.getenv("OVDINO_PSC_NORMALIZE", "0") == "1"
name_to_prompt_path = os.getenv("OVDINO_PSC_NAME2PROMPT") or None

model.language_backbone = L(PseudoLanguageBackbone)(
    text_embed_path=psc_cache,
    name_to_prompt_path=name_to_prompt_path,
    output_dim=psc_dim,
    is_normalize=psc_normalize,
)
# The detector's ClassEmbed projects from text_embed_dim → embed_dim, so
# if you swap to a non-BERT-base encoder (e.g. BiomedCLIP D=512) the model
# side must agree on D before the checkpoint loader runs.
model.text_embed_dim = psc_dim

# Inference template must be "identity" so OV-DINO does not wrap each name
# in an extra "a photo of a {}" template — the cache keys are already the
# resolved prompts.
model.inference_template = "identity"
