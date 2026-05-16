"""Reference eval config: organ-aware OV-DINO on a custom COCO-format dataset.

Switches the model class to ``OVDINOOrganAware`` and wires:
  * ``model.organ_mask_path`` → the ``.pt`` produced by
    ``projects/ovdino/data/build_class_organ_mask.py``
  * an ``organ_id`` extractor on the eval dataset so every record carries
    the organ tag the model needs to mask cross-organ classes

Both paths are env-var driven so the file is reusable across hosts:
  * ``OVDINO_ORGAN_MASK``         absolute path to the mask .pt
  * ``OVDINO_ORGAN_PATH_REGEX``   regex with a named group "organ"
  * ``OVDINO_ORGAN_TO_ID_JSON``   json with {"organ_name": id, ...}
  * ``OVDINO_ORGAN_STRICT``       "0" to allow records without an organ_id
                                  (default: strict — every record must
                                  resolve to a valid organ_id, otherwise
                                  registration raises).

If ``OVDINO_ORGAN_TO_ID_JSON`` is not set, we fall back to the mapping
embedded in the mask file itself (``organ_to_id`` field).
"""

import json
import os
import os.path as osp

from detrex.config import get_config

from .models.ovdino_swin_tiny224_bert_base import model

# Swap the model class to the organ-aware subclass.
from projects.ovdino.modeling.ovdino_organ import OVDINOOrganAware

model._target_ = OVDINOOrganAware

model_root = os.getenv("MODEL_ROOT", "./inits")
init_checkpoint = osp.join(model_root, "./swin", "swin_tiny_patch4_window7_224.pth")

dataloader = get_config("common/data/custom_ovd.py").dataloader
train = get_config("common/train.py").train

train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/ovdino_swin_tiny224_bert_base_eval_custom_organ"
train.device = "cuda"
model.device = train.device

dataloader.evaluator.output_dir = train.output_dir

# ---- organ prior wiring --------------------------------------------------
organ_mask_path = os.getenv("OVDINO_ORGAN_MASK")
if organ_mask_path:
    model.organ_mask_path = organ_mask_path

    # Attach organ_id to each eval record. Done here (config-load time) so
    # the wrap happens BEFORE the loader's first call from the eval loop.
    from projects.ovdino.data.organ_aware_register import (
        attach_organ_ids_to_dataset,
        organ_extractor_from_dominant_class,
        organ_extractor_from_path_regex,
    )
    import torch

    mask_pkg = torch.load(organ_mask_path, weights_only=False, map_location="cpu")
    organ_to_id_default = mask_pkg["organ_to_id"]
    class_to_organ = mask_pkg["mask"].argmax(dim=1).tolist()

    # Resolve organ_to_id (env override > mask default)
    env_lookup_json = os.getenv("OVDINO_ORGAN_TO_ID_JSON")
    organ_to_id = (
        json.loads(env_lookup_json) if env_lookup_json else organ_to_id_default
    )

    path_regex = os.getenv("OVDINO_ORGAN_PATH_REGEX")
    primary = (
        organ_extractor_from_path_regex(path_regex, organ_to_id=organ_to_id)
        if path_regex
        else (lambda _record: None)
    )
    fallback = organ_extractor_from_dominant_class(class_to_organ)

    # Strict by default — every eval record must resolve to a valid
    # organ_id. Opt out (e.g. for partially-tagged datasets) via the env var.
    strict = os.getenv("OVDINO_ORGAN_STRICT", "1") != "0"
    if not strict:
        # Tell the model side to skip masking for organ_id=-1 records,
        # otherwise the strict default on the model would re-raise.
        model.organ_mask_strict = False

    # ``dataloader.test.dataset.names`` is the eval dataset name. OmegaConf
    # attribute access already resolves it to a primitive (str) or a
    # ListConfig (list of strs). Normalize both to a Python list.
    from omegaconf import ListConfig, OmegaConf
    names = dataloader.test.dataset.names
    if isinstance(names, ListConfig):
        names = OmegaConf.to_container(names, resolve=True)
    if isinstance(names, str):
        names = [names]
    for n in names:
        attach_organ_ids_to_dataset(n, primary, fallback=fallback, strict=strict)
