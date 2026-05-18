"""Register TCT_NGC datasets (30-class base + 5-class novel).

Text fed to BERT is the **clinically-descriptive prompt** (Bethesda / PSC
taxonomy) from ``datas/tct_ngc/metadata/category_prompt_name_map_v2.json``,
not the cryptic hyphenated source name (e.g. ``TCT_CCD-hsil_scc_omn``).
Rationale: bare names tokenize into meaningless WordPiece fragments;
OV-DINO's open-vocab design lets us train once with a strong prompt and
A/B different prompts at eval time.

Pitfall handled here: the released annotation JSONs use **two different
id schemes** for the same class names. ``instances_train_dev_base30.json``
keeps the original-32-class gappy ids (0..16, 18, 19, 23, 31..40);
``instances_test_base_clean_dev30.json`` uses contiguous 0..29. We can't
use a single ``{id: prompt}`` mapping for both — instead we peek each
JSON's ``categories`` array at register time and look up the prompt **by
class name string** (via ``current_to_prompt_name`` in the metadata).

The 30-class scheme drops 3 Urine sub-classes from the 32-class source
and merges them into ``Urine-NHGUC`` — see ``ovdino/scripts/build_tct_ngc_base30.py``.
The 5-class novel set is the literal categories in ``instances_test_novel.json``.
"""
import json
import os

from .custom_ovd import register_custom_ovd_instances


def _find_texts_dir() -> str:
    override = os.getenv("TCT_NGC_TEXTS_DIR")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(here, "data", "texts")
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError(
        "Could not locate data/texts/ from "
        f"{os.path.dirname(os.path.abspath(__file__))}"
    )


_TEXTS_DIR = _find_texts_dir()


def _load_name_to_prompt(root: str) -> dict:
    """Load ``{bare_class_name: descriptive_prompt}`` from dataset metadata.

    Adds a fallback entry for the synthetic ``Urine-NHGUC`` class created
    by the 32->30 merge — it doesn't exist in the source ``current_to_prompt_name``
    (which has the three pre-merge Urine subnames).
    """
    meta_path = os.path.join(
        root, "tct_ngc", "metadata", "category_prompt_name_map_v2.json"
    )
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    mapping = dict(meta.get("current_to_prompt_name", {}))
    # Synthetic post-merge name: 3 Urine NHGUC-variant classes collapsed to one.
    # Use the clean NHGUC prompt without the per-source-label suffix.
    mapping["Urine-NHGUC"] = (
        "Urinary cytology - Negative for high-grade urothelial carcinoma (NHGUC)"
    )
    return mapping


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_NAME_TO_PROMPT = _load_name_to_prompt(_root)


def _load_base30_keys() -> list:
    """Canonical 30-class name list (post-merge) — used to give the random-embed
    ablation a stable ``random_vocab_names`` ordering, and to assert the JSON
    contains exactly these names.
    """
    path = os.path.join(_TEXTS_DIR, "tct_ngc_class_keys_base30.json")
    with open(path) as f:
        keys = json.load(f)
    names = list(keys.keys())
    assert len(names) == 30, f"expected 30 keys in {path}, got {len(names)}"
    return names


_BASE30_NAMES = _load_base30_keys()

# Novel 5 — verbatim from instances_test_novel.json `categories[].name`.
_NOVEL5_NAMES = [
    "respiratory tract-adenocarcinoma",
    "Serous effusion-Ovarian cancer",
    "respiratory tract-Squamous cell carcinoma",
    "Serous effusion-Breast cancer",
    "Thyroid gland-MTC",
]


def _resolve_prompt(bare_name: str) -> str:
    prompt = _NAME_TO_PROMPT.get(bare_name)
    if prompt is None:
        # Should never happen — every canonical name is covered by metadata
        # + the Urine-NHGUC fallback. Surface loudly if we missed a case.
        raise KeyError(
            f"No prompt mapping for class {bare_name!r}; check "
            "current_to_prompt_name in category_prompt_name_map_v2.json"
        )
    return prompt


# Exposed for variant configs (e.g. random-embed ablation): the canonical
# prompt forms in canonical order. random_vocab_names = these prompts.
TCT_NGC_BASE_CATEGORIES = [
    {"name": _resolve_prompt(n), "id": i + 1} for i, n in enumerate(_BASE30_NAMES)
]
TCT_NGC_NOVEL_CATEGORIES = [
    {"name": _resolve_prompt(n), "id": i + 1} for i, n in enumerate(_NOVEL5_NAMES)
]
NUM_BASE_CATEGORY = len(TCT_NGC_BASE_CATEGORIES)
NUM_NOVEL_CATEGORY = len(TCT_NGC_NOVEL_CATEGORIES)


def _build_id_mapping(json_path: str) -> dict:
    """Peek the COCO JSON's `categories` and return ``{cat_id: prompt}``
    keyed by whatever id scheme that file happens to use. Lookup is by
    class-name string so it works for both gappy and contiguous schemes.
    """
    full_path = json_path if os.path.isabs(json_path) else os.path.join(_root, json_path)
    with open(full_path) as f:
        d = json.load(f)
    cats = d.get("categories", [])
    return {c["id"]: _resolve_prompt(c["name"]) for c in cats}


def _get_meta(categories):
    # Don't pre-set thing_dataset_id_to_contiguous_id / thing_classes here —
    # load_custom_json (in custom_ovd.py) populates them from the actual JSON
    # at first DatasetCatalog.get() call, AND it honors the name_mapping arg
    # to rewrite the names. MetadataCatalog refuses to overwrite an existing
    # value, so leaving these unset is the only safe option.
    return {}


_PREDEFINED_SPLITS = {
    # key: (image_root, json_file, num_sampled_classes, template, categories)
    # name_mapping is computed per-split at register time via _build_id_mapping.
    "tct_ngc_train_base_ovd_unipro": (
        "tct_ngc/images",
        "tct_ngc/annotations/instances_train_dev_base30.json",
        NUM_BASE_CATEGORY,
        "full",
        TCT_NGC_BASE_CATEGORIES,
    ),
    "tct_ngc_val_base_ovd": (
        "tct_ngc/images",
        "tct_ngc/annotations/instances_val_dev_base30.json",
        NUM_BASE_CATEGORY,
        "identity",
        TCT_NGC_BASE_CATEGORIES,
    ),
    "tct_ngc_test_base_ovd": (
        "tct_ngc/images",
        "tct_ngc/annotations/instances_test_base_clean_dev30.json",
        NUM_BASE_CATEGORY,
        "identity",
        TCT_NGC_BASE_CATEGORIES,
    ),
    "tct_ngc_test_novel_ovd": (
        "tct_ngc/images",
        "tct_ngc/annotations/instances_test_novel.json",
        NUM_NOVEL_CATEGORY,
        "identity",
        TCT_NGC_NOVEL_CATEGORIES,
    ),
}


def register_all_tct_ngc_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
        template,
        categories,
    ) in _PREDEFINED_SPLITS.items():
        json_path_abs = os.path.join(root, json_file)
        # Build per-split id->prompt mapping by reading the JSON's categories.
        # This handles the gappy-vs-contiguous id-scheme split between
        # train_dev_base30 and test_base_clean_dev30.
        name_mapping = _build_id_mapping(json_path_abs)
        register_custom_ovd_instances(
            key,
            _get_meta(categories),
            json_path_abs,
            os.path.join(root, image_root),
            num_sampled_classes,
            template=template,
            test_mode="val" in key or "test" in key,
            name_mapping=name_mapping,
        )


register_all_tct_ngc_instances(_root)
