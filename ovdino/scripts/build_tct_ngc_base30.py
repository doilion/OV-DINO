#!/usr/bin/env python3
"""Build 30-class TCT_NGC train/val annotation JSONs from 32-class source.

Merges three Urine sub-classes into a single ``Urine-NHGUC`` to align the
training data with the canonical 30-class scheme used by
``instances_test_base_clean_dev30.json`` and ``data/texts/tct_ngc_class_*_base30.json``.

Merge rule (kept consistent with the released dev30 eval file):
    category_id 17 (Urine-Negative)             -> 16
    category_id 20 (Urine-Negative Degeneration) -> 16
    rename id 16 "Urine-NILM" -> "Urine-NHGUC"

Original category_ids are preserved (gappy) — detectron2's load_custom_json
remaps them to contiguous [0, 30) at load time.

Usage:
    python ovdino/scripts/build_tct_ngc_base30.py \\
        --in-dir /root/commonfile/TCT_NGC/annotations \\
        --out-dir /root/commonfile/TCT_NGC/annotations
"""
import argparse
import json
import os
from pathlib import Path

DROP_IDS = {17, 20}
MERGE_INTO = 16
NEW_NAME = "Urine-NHGUC"


def merge_one(src_path: Path, dst_path: Path) -> dict:
    with src_path.open() as f:
        data = json.load(f)

    src_cats = data["categories"]
    src_n_anns = len(data["annotations"])

    # Rewrite categories: drop {17,20}, rename 16 → Urine-NHGUC
    new_cats = []
    for c in src_cats:
        if c["id"] in DROP_IDS:
            continue
        if c["id"] == MERGE_INTO:
            new_cats.append({**c, "name": NEW_NAME})
        else:
            new_cats.append(c)
    assert len(new_cats) == 30, f"expected 30 cats, got {len(new_cats)}"

    # Rewrite annotations: redirect dropped ids to MERGE_INTO
    merge_count = 0
    for a in data["annotations"]:
        if a["category_id"] in DROP_IDS:
            a["category_id"] = MERGE_INTO
            merge_count += 1

    assert len(data["annotations"]) == src_n_anns, "annotation count must be preserved"

    data["categories"] = new_cats

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w") as f:
        json.dump(data, f)

    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "n_images": len(data["images"]),
        "n_anns": len(data["annotations"]),
        "n_cats": len(new_cats),
        "n_merged": merge_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/root/commonfile/TCT_NGC/annotations")
    ap.add_argument("--out-dir", default="/root/commonfile/TCT_NGC/annotations")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    targets = [
        ("instances_train_dev.json", "instances_train_dev_base30.json"),
        ("instances_val_dev.json", "instances_val_dev_base30.json"),
    ]

    results = []
    for src_name, dst_name in targets:
        r = merge_one(in_dir / src_name, out_dir / dst_name)
        results.append(r)
        print(
            f"  {src_name} → {dst_name}: "
            f"imgs={r['n_images']} anns={r['n_anns']} "
            f"cats={r['n_cats']} (merged {r['n_merged']} anns into id={MERGE_INTO})"
        )

    # Traceability sidecar
    remap = {
        "dropped_category_ids": sorted(DROP_IDS),
        "merged_into_id": MERGE_INTO,
        "rename": {str(MERGE_INTO): NEW_NAME},
        "outputs": [r["dst"] for r in results],
    }
    remap_path = out_dir / "base30_remap.json"
    with remap_path.open("w") as f:
        json.dump(remap, f, indent=2)
    print(f"\n  Wrote remap manifest: {remap_path}")


if __name__ == "__main__":
    main()
