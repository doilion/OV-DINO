"""Attach ``organ_id`` to a registered detectron2 dataset's records.

``OVDINOOrganAware`` reads ``batched_inputs[b]["organ_id"]`` to build the
per-image organ mask. The standard detrex loader (``custom_ovd.load_custom_ovd``)
does not produce that key, so we add it here without modifying the loader.

Two extraction modes are supported:

1. **By image filename** — pass a regex (or any callable) that maps
   ``record["file_name"]`` to an ``int`` organ_id.
2. **By dominant ground-truth class** — pass a class-id → organ-id lookup
   (e.g. ``mask.argmax(dim=1)`` of the file built by
   ``build_class_organ_mask.py``) and we pick the organ of the majority GT
   annotation in the record. Useful when filenames carry no organ signal.

Both modes can be combined: filename takes precedence; the GT mode is the
fallback when the regex doesn't match.

Usage in a LazyConfig::

    from projects.ovdino.data.organ_aware_register import (
        attach_organ_ids_to_dataset, organ_extractor_from_path_regex,
    )

    # After custom_ovd's auto-registration, wrap the eval dataset:
    attach_organ_ids_to_dataset(
        "custom_val_ovd_unipro",
        extractor=organ_extractor_from_path_regex(
            r".*/(?P<organ>[^/]+)/[^/]+\\.(jpg|png|jpeg)$",
            organ_to_id={"thyroid": 1, "urine": 2, ...},
        ),
    )

The wrap is idempotent — calling it twice on the same dataset name is safe.
"""

import logging
import os
import re
from collections import Counter
from typing import Callable, Dict, List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog


_WRAPPED_FLAG_ATTR = "_ovdino_organ_id_attached"
_ORIGINAL_LOADER_ATTR = "_ovdino_organ_id_original_loader"
_logger = logging.getLogger(__name__)


def organ_extractor_from_path_regex(
    pattern: str,
    organ_to_id: Dict[str, int],
    group: str = "organ",
    case_insensitive: bool = True,
) -> Callable[[dict], Optional[int]]:
    """Build an extractor that reads organ_id from ``record["file_name"]``.

    ``pattern`` is a regex with a named group (``"organ"`` by default); the
    matched substring is looked up in ``organ_to_id``. Returns ``None`` on a
    miss so a fallback extractor can be tried.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    rx = re.compile(pattern, flags)
    if case_insensitive:
        # Catch silent collisions like {"Thyroid": 1, "thyroid": 2} that
        # would otherwise produce nondeterministic dict iteration order.
        lower_pairs = [(k.lower(), v) for k, v in organ_to_id.items()]
        seen: Dict[str, int] = {}
        for k, v in lower_pairs:
            if k in seen and seen[k] != v:
                raise ValueError(
                    f"organ_to_id has case-insensitive duplicate '{k}' "
                    f"mapping to both {seen[k]} and {v}; pass "
                    "case_insensitive=False or deduplicate the mapping."
                )
            seen[k] = v
        lookup = seen
    else:
        lookup = dict(organ_to_id)

    def _extract(record: dict) -> Optional[int]:
        path = record.get("file_name", "")
        m = rx.search(path)
        if not m:
            return None
        key = m.group(group)
        if case_insensitive:
            key = key.lower()
        return lookup.get(key)

    return _extract


def organ_extractor_from_dominant_class(
    class_to_organ: List[int],
) -> Callable[[dict], Optional[int]]:
    """Build an extractor that picks the organ of the majority GT class.

    ``class_to_organ`` is a list of length ``num_classes`` whose ``[i]``-th
    element is the organ_id of contiguous class ``i``. Get it from the mask
    file via ``mask.argmax(dim=1).tolist()``.
    """

    def _extract(record: dict) -> Optional[int]:
        annos = record.get("annotations") or []
        if not annos:
            return None
        # ``annotations`` may carry COCO category_ids (pre-id_map) or
        # contiguous ids (post). detrex's loader applies id_map in-place, so
        # by the time records land in DatasetCatalog the ids are contiguous.
        cls_ids = [a["category_id"] for a in annos if "category_id" in a]
        if not cls_ids:
            return None
        most_common, _ = Counter(cls_ids).most_common(1)[0]
        if not (0 <= most_common < len(class_to_organ)):
            return None
        return int(class_to_organ[most_common])

    return _extract


def attach_organ_ids_to_dataset(
    dataset_name: str,
    extractor: Callable[[dict], Optional[int]],
    fallback: Optional[Callable[[dict], Optional[int]]] = None,
    strict: bool = True,
    force: bool = False,
) -> None:
    """Wrap a registered dataset so every record carries ``organ_id``.

    Args:
        dataset_name: name already registered in ``DatasetCatalog``.
        extractor: primary extractor (e.g. filename-based).
        fallback: optional secondary extractor tried when primary returns
            ``None`` (e.g. dominant-class).
        strict: if ``True``, raises when a record yields no organ_id from
            either extractor. If ``False``, attaches ``organ_id=-1`` and
            logs a warning — combine with ``organ_mask_strict=False`` on
            the model to let those samples skip masking.
        force: if ``True``, re-wrap even when the dataset has already been
            wrapped in this process. The wrapper restores the truly-original
            loader before installing the new one, so chained wraps don't
            stack. Use when loading a different organ-aware config in the
            same Python session (e.g. hydra sweeps).
    """
    if dataset_name not in DatasetCatalog.list():
        raise KeyError(
            f"'{dataset_name}' is not registered; import the dataset module "
            "(or call its register_all_*) before wrapping it."
        )

    meta = MetadataCatalog.get(dataset_name)
    already_wrapped = getattr(meta, _WRAPPED_FLAG_ATTR, False)
    if already_wrapped and not force:
        _logger.info(
            "[organ_aware_register] '%s' already wrapped, skipping "
            "(pass force=True to re-wrap with a new extractor)",
            dataset_name,
        )
        return

    # DatasetCatalog is a UserDict[str, Callable]. If this dataset has been
    # wrapped before, restore the truly-original loader so wrappers don't
    # stack; otherwise capture the original now.
    if already_wrapped:
        original_loader = getattr(meta, _ORIGINAL_LOADER_ATTR)
    else:
        original_loader = DatasetCatalog[dataset_name]
        setattr(meta, _ORIGINAL_LOADER_ATTR, original_loader)
    DatasetCatalog.remove(dataset_name)

    def wrapped_loader():
        records = original_loader()
        missing = 0
        for r in records:
            organ_id = extractor(r)
            if organ_id is None and fallback is not None:
                organ_id = fallback(r)
            if organ_id is None:
                if strict:
                    raise RuntimeError(
                        f"organ_aware_register: could not resolve organ_id for "
                        f"record file_name={r.get('file_name')!r}. Either fix "
                        "your extractor / taxonomy, or pass strict=False to "
                        "tolerate misses."
                    )
                organ_id = -1
                missing += 1
            r["organ_id"] = int(organ_id)
        if missing:
            _logger.warning(
                "[organ_aware_register] %s: %d/%d records had no organ_id "
                "(strict=False), tagging with -1",
                dataset_name,
                missing,
                len(records),
            )
        return records

    DatasetCatalog.register(dataset_name, wrapped_loader)
    setattr(meta, _WRAPPED_FLAG_ATTR, True)
    _logger.info(
        "[organ_aware_register] wrapped '%s' with organ_id extractor", dataset_name
    )
