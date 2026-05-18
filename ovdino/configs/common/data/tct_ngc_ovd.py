import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detrex.evaluation import ExcludeClassCOCOEvaluator
from detrex.data import DetrDatasetMapper
from omegaconf import OmegaConf

dataloader = OmegaConf.create()

# Fixed 640 short edge + max 1024 long edge — chosen to fit batch_size=2/GPU
# on 2080Ti 11GB. Multi-scale (480-800, max 1333) only fits bs=1/GPU.
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="tct_ngc_train_base_ovd_unipro"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=640,
                max_size=1024,
            ),
        ],
        augmentation_with_crop=None,
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    # bs=2/GPU @ 640+1024 peaked at 10.9GB on 11GB 2080Ti (verified) due to
    # wide WSI source crops (~4000x2800) — transient OOM on bad-luck batches.
    # 1/GPU stays safe; raise if a future config uses smaller max_size.
    total_batch_size=8,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="tct_ngc_test_base_ovd", filter_empty=False
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=640,
                max_size=1024,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

# Exclude the 30-class scheme's negative / non-diagnostic categories from mAP.
# These names are the **prompt forms** (post name_mapping in
# register_tct_ngc_ovd.py) — they must match `thing_classes` byte-for-byte
# (case- and whitespace-sensitive). The corresponding bare source labels are
# noted in the comments for traceability.
dataloader.evaluator = L(ExcludeClassCOCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    exclude_class_names=[
        # id 7  | bare: Serous effusion-Negative samples
        "Serous fluid cytology - Negative for malignancy (NFM)",
        # id 14 | bare: Thyroid gland-Negative samples
        "Thyroid cytopathology - Benign (Bethesda II) [source label: Negative samples]",
        # id 16 | bare (merged): Urine-NHGUC (merge target of NILM/Negative/Neg-Degen)
        "Urinary cytology - Negative for high-grade urothelial carcinoma (NHGUC)",
        # id 31 | bare: TCT_CCD-normal
        "Cervical cytology - Negative for intraepithelial lesion or malignancy (NILM)",
        # id 4  | bare: respiratory tract-Impurity
        "Respiratory tract cytology - Contaminant/debris",
        # id 11 | bare: Thyroid gland-NS
        "Thyroid cytopathology - Benign (Bethesda II) [source label: NS]",
    ],
)
