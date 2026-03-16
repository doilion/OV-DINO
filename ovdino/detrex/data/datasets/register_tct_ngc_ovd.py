import os

from .custom_ovd import register_custom_ovd_instances


TCT_NGC_BASE_CATEGORIES = [
    {"name": "normal", "id": 1},
    {"name": "ascus", "id": 2},
    {"name": "asch", "id": 3},
    {"name": "lsil", "id": 4},
    {"name": "agc_adenocarcinoma_em", "id": 5},
    {"name": "vaginalis", "id": 6},
    {"name": "dysbacteriosis_herpes_act", "id": 7},
    {"name": "ec", "id": 8},
    {"name": "Serous effusion-Negative samples", "id": 9},
    {"name": "Serous effusion-Diseased cells", "id": 10},
    {"name": "Serous effusion-Breast cancer", "id": 11},
    {"name": "Thyroid gland-Papillary cancer", "id": 12},
    {"name": "Thyroid gland-Negative samples", "id": 13},
    {"name": "Thyroid gland-Suspicious for Malignancy", "id": 14},
    {"name": "Urine-Negative", "id": 15},
    {"name": "Urine-SHGUC", "id": 16},
    {"name": "Urine-AUC", "id": 17},
    {"name": "respiratory tract-Negative samples", "id": 18},
    {"name": "respiratory tract-Diseased cells", "id": 19},
    {"name": "respiratory tract-adenocarcinoma", "id": 20},
]

NUM_BASE_CATEGORY = len(TCT_NGC_BASE_CATEGORIES)


def _get_tct_ngc_base_meta():
    thing_ids = [k["id"] for k in TCT_NGC_BASE_CATEGORIES]
    assert len(thing_ids) == NUM_BASE_CATEGORY, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in TCT_NGC_BASE_CATEGORIES]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS = {
    # image_root, json_file, num_sampled_classes, template
    "tct_ngc_train_base_ovd_unipro": (
        "tct_ngc",
        "tct_ngc/annotations/train_base_v2.json",
        NUM_BASE_CATEGORY,
        "full",
    ),
    "tct_ngc_val_base_ovd": (
        "tct_ngc",
        "tct_ngc/annotations/test_base_v2.json",
        NUM_BASE_CATEGORY,
        "identity",
    ),
    "tct_ngc_test_base_ovd": (
        "tct_ngc",
        "tct_ngc/annotations/test_base_v2.json",
        NUM_BASE_CATEGORY,
        "identity",
    ),
}


def register_all_tct_ngc_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
        template,
    ) in _PREDEFINED_SPLITS.items():
        register_custom_ovd_instances(
            key,
            _get_tct_ngc_base_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            num_sampled_classes,
            template=template,
            test_mode="val" in key or "test" in key,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_tct_ngc_instances(_root)
