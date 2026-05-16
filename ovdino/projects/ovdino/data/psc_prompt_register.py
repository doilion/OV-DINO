"""Swap bare class names for long PSC prompts at DatasetCatalog load time.

Pure string substitution — no model change, no embedding cache. The
existing ``BERTEncoder`` simply encodes the longer descriptive sentence
instead of the bare class name. Use this when you want an A/B
comparison of ``OV-DINO + bare names`` vs ``OV-DINO + PSC prompts`` with
everything else (model, weights, schedule) held constant.

Mirrors the wrapper pattern of ``organ_aware_register``: pops the
original loader thunk out of ``DatasetCatalog`` and re-registers a
wrapper that rewrites every record's ``category_names`` field.

Usage in a LazyConfig::

    from projects.ovdino.data.psc_prompt_register import (
        attach_psc_prompts_to_dataset,
    )

    attach_psc_prompts_to_dataset(
        "custom_val_ovd_unipro",
        name_to_prompt_path="data/texts/tct_ngc_class_prompts_base30.json",
    )

The JSON must be a flat ``{class_name: prompt_string}`` map keyed by the
**cleaned** class names that ``detrex.data.datasets.custom_ovd`` produces
(post ``clean_words_or_phrase``). The 30-class TCT_NGC file shipped in
``data/texts/`` is already in that shape.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from detectron2.data import DatasetCatalog, MetadataCatalog

_logger = logging.getLogger(__name__)
_WRAPPED_FLAG_ATTR = "_ovdino_psc_prompts_attached"
_ORIGINAL_LOADER_ATTR = "_ovdino_psc_prompts_original_loader"


def attach_psc_prompts_to_dataset(
    dataset_name: str,
    name_to_prompt_path: str,
    strict: bool = True,
    force: bool = False,
) -> None:
    """Substitute each record's ``category_names`` with PSC prompts.

    Args:
        dataset_name: name already registered in ``DatasetCatalog``.
        name_to_prompt_path: JSON file holding ``{class_name: prompt}``.
        strict: if ``True``, raises when a class in the dataset has no
            entry in the JSON. If ``False``, keeps the original bare
            name for the missing class and logs a warning (lets you
            ablate "swap N of M classes" if needed).
        force: re-wrap even if already wrapped this session. The wrapper
            restores the truly-original loader before installing the new
            one so wrappers don't stack.
    """
    if dataset_name not in DatasetCatalog.list():
        raise KeyError(
            f"'{dataset_name}' is not registered; import the dataset module "
            "(or call its register_all_*) before wrapping it."
        )

    table = json.loads(Path(name_to_prompt_path).read_text())
    if not isinstance(table, dict):
        raise ValueError(
            f"{name_to_prompt_path}: expected a JSON object {{class_name: "
            f"prompt}}, got {type(table).__name__}."
        )

    meta = MetadataCatalog.get(dataset_name)
    already_wrapped = getattr(meta, _WRAPPED_FLAG_ATTR, False)
    if already_wrapped and not force:
        _logger.info(
            "[psc_prompt_register] '%s' already wrapped, skipping "
            "(pass force=True to re-wrap with a new prompt map)",
            dataset_name,
        )
        return

    if already_wrapped:
        original_loader = getattr(meta, _ORIGINAL_LOADER_ATTR)
    else:
        original_loader = DatasetCatalog[dataset_name]
        setattr(meta, _ORIGINAL_LOADER_ATTR, original_loader)
    DatasetCatalog.remove(dataset_name)

    def wrapped_loader():
        records = original_loader()
        missing: set = set()
        for r in records:
            names = r.get("category_names")
            if not names:
                continue
            substituted = []
            for nm in names:
                if nm in table:
                    substituted.append(table[nm])
                else:
                    missing.add(nm)
                    substituted.append(nm)
            r["category_names"] = substituted
        if missing:
            msg = (
                f"[psc_prompt_register] {dataset_name}: {len(missing)} "
                f"class name(s) had no PSC entry: {sorted(missing)[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
            if strict:
                raise RuntimeError(
                    msg + "\nPass strict=False to keep bare names for the "
                    "misses, or extend the JSON to cover the missing classes."
                )
            _logger.warning(msg)
        # NOTE: we intentionally do NOT mutate meta.thing_classes here.
        # detectron2's MetadataCatalog forbids reassigning it, and keeping
        # the bare class names on the metadata side means the per-class AP
        # table stays human-readable across the bare-vs-PSC A/B (the model
        # sees the long prompt via category_names; the evaluator labels
        # rows by thing_classes, which stays bare).
        return records

    DatasetCatalog.register(dataset_name, wrapped_loader)
    setattr(meta, _WRAPPED_FLAG_ATTR, True)
    _logger.info(
        "[psc_prompt_register] wrapped '%s' with %d PSC prompts from %s",
        dataset_name,
        len(table),
        name_to_prompt_path,
    )
