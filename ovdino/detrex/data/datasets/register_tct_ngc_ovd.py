import os

from .custom_ovd import register_custom_ovd_instances


# Base categories (20): descriptive English names for BERT text encoder
TCT_NGC_BASE_CATEGORIES = [
    {"name": "cervical normal cells", "id": 1},
    {"name": "cervical atypical squamous cells of undetermined significance", "id": 2},
    {"name": "cervical atypical squamous cells cannot exclude high-grade lesion", "id": 3},
    {"name": "cervical low-grade squamous intraepithelial lesion", "id": 4},
    {"name": "cervical atypical glandular cells and adenocarcinoma with endometrial origin", "id": 5},
    {"name": "cervical trichomonas vaginalis infection", "id": 6},
    {"name": "cervical dysbacteriosis with herpes and actinomyces", "id": 7},
    {"name": "cervical endocervical cells", "id": 8},
    {"name": "serous effusion negative samples", "id": 9},
    {"name": "serous effusion diseased cells", "id": 10},
    {"name": "serous effusion breast cancer cells", "id": 11},
    {"name": "thyroid gland papillary cancer", "id": 12},
    {"name": "thyroid gland negative samples", "id": 13},
    {"name": "thyroid gland suspicious for malignancy", "id": 14},
    {"name": "urine cytology negative samples", "id": 15},
    {"name": "urine suspicious for high-grade urothelial carcinoma", "id": 16},
    {"name": "urine atypical urothelial cells", "id": 17},
    {"name": "respiratory tract negative samples", "id": 18},
    {"name": "respiratory tract diseased cells", "id": 19},
    {"name": "respiratory tract adenocarcinoma", "id": 20},
]

# Novel categories (11): descriptive English names for BERT text encoder
TCT_NGC_NOVEL_CATEGORIES = [
    {"name": "cervical high-grade squamous intraepithelial lesion and squamous cell carcinoma", "id": 1},
    {"name": "cervical candida infection", "id": 2},
    {"name": "serous effusion ovarian cancer cells", "id": 3},
    {"name": "serous effusion adenocarcinoma cells", "id": 4},
    {"name": "thyroid gland suspicious for papillary cancer", "id": 5},
    {"name": "thyroid gland atypia of undetermined significance", "id": 6},
    {"name": "thyroid gland malignant tumour", "id": 7},
    {"name": "thyroid gland non-diagnostic specimen", "id": 8},
    {"name": "urine high-grade urothelial carcinoma", "id": 9},
    {"name": "respiratory tract squamous cell carcinoma", "id": 10},
    {"name": "respiratory tract small cell carcinoma", "id": 11},
]

NUM_BASE_CATEGORY = len(TCT_NGC_BASE_CATEGORIES)
NUM_NOVEL_CATEGORY = len(TCT_NGC_NOVEL_CATEGORIES)

# Name mappings: annotation category_id -> descriptive name
BASE_NAME_MAPPING = {c["id"]: c["name"] for c in TCT_NGC_BASE_CATEGORIES}
NOVEL_NAME_MAPPING = {c["id"]: c["name"] for c in TCT_NGC_NOVEL_CATEGORIES}


def _get_meta(categories):
    thing_ids = [k["id"] for k in categories]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in categories]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS = {
    # key: (image_root, json_file, num_sampled_classes, template, categories, name_mapping)
    # --- Base splits ---
    "tct_ngc_train_base_ovd_unipro": (
        "tct_ngc",
        "tct_ngc/annotations/train_base_v2.json",
        NUM_BASE_CATEGORY,
        "full",
        TCT_NGC_BASE_CATEGORIES,
        BASE_NAME_MAPPING,
    ),
    "tct_ngc_val_base_ovd": (
        "tct_ngc",
        "tct_ngc/annotations/test_base_v2.json",
        NUM_BASE_CATEGORY,
        "identity",
        TCT_NGC_BASE_CATEGORIES,
        BASE_NAME_MAPPING,
    ),
    "tct_ngc_test_base_ovd": (
        "tct_ngc",
        "tct_ngc/annotations/test_base_v2.json",
        NUM_BASE_CATEGORY,
        "identity",
        TCT_NGC_BASE_CATEGORIES,
        BASE_NAME_MAPPING,
    ),
    # --- Novel splits ---
    "tct_ngc_train_novel_ovd_unipro": (
        "tct_ngc",
        "tct_ngc/annotations/train_novel_v2.json",
        NUM_NOVEL_CATEGORY,
        "full",
        TCT_NGC_NOVEL_CATEGORIES,
        NOVEL_NAME_MAPPING,
    ),
    "tct_ngc_test_novel_ovd": (
        "tct_ngc",
        "tct_ngc/annotations/test_novel_v2.json",
        NUM_NOVEL_CATEGORY,
        "identity",
        TCT_NGC_NOVEL_CATEGORIES,
        NOVEL_NAME_MAPPING,
    ),
}


def register_all_tct_ngc_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
        template,
        categories,
        name_mapping,
    ) in _PREDEFINED_SPLITS.items():
        register_custom_ovd_instances(
            key,
            _get_meta(categories),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            num_sampled_classes,
            template=template,
            test_mode="val" in key or "test" in key,
            name_mapping=name_mapping,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_tct_ngc_instances(_root)
