"""Build a class×organ valid mask for OV-DINO organ-conditional inference.

Port of WeDetect's ``tools/build_class_organ_mask.py``. The output file is
fully interchangeable with WeDetect's, because both projects sort COCO
categories by ascending ``id`` to derive ``class_names`` — which is exactly
the ordering ``detrex.data.datasets.custom_ovd`` uses to populate
``MetadataCatalog.thing_classes``. So a mask built here against the same
ann file works for OV-DINO's contiguous-index space without remapping.

Output package (loaded by ``OVDINOOrganAware``)::

    {
        'class_names': [str, ...],    # ordered by ascending COCO category_id
        'class_ids':   [int, ...],    # original COCO category_id
        'organ_names': [str, ...],
        'organ_to_id': {name: int},
        'mask':        FloatTensor[C, O],   # mask[c, o] = 1 iff classes[c].organ_id == o
    }

Usage::

    python projects/ovdino/data/build_class_organ_mask.py \\
        --ann datas/custom/annotations/test.json \\
        --taxonomy data/texts/tct_ngc_taxonomy.json \\
        --out data/texts/tct_ngc_class_organ_mask.pt

The taxonomy JSON is the same one WeDetect uses; format::

    {
        "organs": ["respiratory tract", "Thyroid gland", "Urine", ...],
        "organ_to_id": {"respiratory tract": 0, "Thyroid gland": 1, ...},
        "classes": {
            "Thyroid gland-PTC": {"organ_id": 1},
            ...
        }
    }
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, type=Path,
                   help="COCO-format ann json defining the test class list")
    p.add_argument("--taxonomy", required=True, type=Path,
                   help="taxonomy json mapping class name → organ_id")
    p.add_argument("--out", required=True, type=Path, help="output .pt path")
    args = p.parse_args()

    ann = json.loads(args.ann.read_text())
    tax = json.loads(args.taxonomy.read_text())

    organs = tax["organs"]
    organ_to_id = tax["organ_to_id"]
    classes_meta = tax["classes"]
    n_organs = len(organs)

    sorted_cats = sorted(ann["categories"], key=lambda c: c["id"])
    class_names = [c["name"] for c in sorted_cats]
    class_ids = [c["id"] for c in sorted_cats]
    n_classes = len(class_names)

    mask = torch.zeros(n_classes, n_organs, dtype=torch.float32)
    unmapped = []
    for i, name in enumerate(class_names):
        if name not in classes_meta:
            unmapped.append(name)
            continue
        organ_id = classes_meta[name].get("organ_id")
        if organ_id is None:
            unmapped.append(name)
            continue
        mask[i, organ_id] = 1.0

    if unmapped:
        raise SystemExit(
            f"taxonomy is missing organ_id for {len(unmapped)} classes:\n  "
            + "\n  ".join(unmapped)
        )

    assert torch.all(mask.sum(dim=1) == 1.0), (
        "each class must map to exactly one organ"
    )

    out = {
        "class_names": class_names,
        "class_ids": class_ids,
        "organ_names": organs,
        "organ_to_id": organ_to_id,
        "mask": mask,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)

    print(f"saved: {args.out}")
    print(f"  C={n_classes} classes  O={n_organs} organs  shape={tuple(mask.shape)}")
    per_organ = mask.sum(dim=0).int().tolist()
    for name, n in zip(organs, per_organ):
        print(f"  {name:24s}  {n} classes")


if __name__ == "__main__":
    main()
