#!/usr/bin/env python
"""Evaluate an OV-DINO checkpoint while excluding "negative" classes.

Mirrors WeDetect's ``test_exclude_negative.py``: at load time, swap the
LazyConfig's ``dataloader.evaluator`` for an exclude-class variant that

* drops predictions belonging to the excluded class set, and
* restricts the underlying COCOeval / LVISEval to the remaining categories
  for mAP averaging and the per-class AP table.

This keeps the original eval config intact (so it can still be used for
unfiltered runs) and only diverges via this script.

Run via ``scripts/eval_exclude_negative.sh`` to get the same
``DETECTRON2_DATASETS`` / ``HF_HOME`` / ``MODEL_ROOT`` plumbing as
``scripts/eval.sh``.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

# Make ``projects.*`` importable when invoked directly. ``train_net.py`` does
# the same trick.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from detectron2.config import LazyCall as L  # noqa: E402
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detrex.modeling import ema
from detrex.utils import setup_dist_args

# train_net.py lives next door; reuse its do_test (carries EMA, distributed glue).
from train_net import do_test  # noqa: E402

from projects.ovdino.evaluation import (  # noqa: E402
    ExcludeClassCOCOEvaluator,
    ExcludeClassLVISFixedAPEvaluator,
)


# Default negative-class list copied from WeDetect's ``test_exclude_negative.py``.
# Override with ``--exclude-class-names`` for custom datasets.
DEFAULT_NEGATIVE_CLASS_NAMES: List[str] = [
    "respiratory tract-Impurity",
    "Serous effusion-Negative samples",
    "Thyroid gland-Negative samples",
    "Urine-NILM",
    "Urine-Negative",
    "Urine-Negative Degeneration",
    "TCT_CCD-normal",
]


def _swap_evaluator(cfg, exclude_class_names: List[str], logger: logging.Logger):
    """Replace ``cfg.dataloader.evaluator`` with the exclude-class variant.

    Preserves all existing kwargs on the original evaluator (``dataset_name``,
    ``output_dir``, ``max_dets_per_image``, ``topk`` for LVIS, etc.) and tacks
    ``exclude_class_names`` on top.
    """
    if not hasattr(cfg, "dataloader") or not hasattr(cfg.dataloader, "evaluator"):
        raise RuntimeError(
            "Config has no dataloader.evaluator to swap; "
            "ensure the config inherits from common/data/<dataset>_ovd.py."
        )

    orig = cfg.dataloader.evaluator
    target_str = getattr(orig, "_target_", "")
    logger.info("[eval_exclude_negative] original evaluator: %s", target_str)

    # Copy non-meta kwargs from the original LazyCall.
    kwargs = {k: v for k, v in dict(orig).items() if not k.startswith("_")}
    kwargs["exclude_class_names"] = list(exclude_class_names)

    if target_str.endswith("COCOEvaluator"):
        cfg.dataloader.evaluator = L(ExcludeClassCOCOEvaluator)(**kwargs)
    elif target_str.endswith("LVISFixedAPEvaluator"):
        cfg.dataloader.evaluator = L(ExcludeClassLVISFixedAPEvaluator)(**kwargs)
    else:
        raise RuntimeError(
            f"Unsupported evaluator target '{target_str}'. Only COCOEvaluator "
            "and LVISFixedAPEvaluator are wrapped by this script."
        )
    logger.info(
        "[eval_exclude_negative] swapped to %s, excluding %d class(es)",
        cfg.dataloader.evaluator._target_,
        len(exclude_class_names),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = default_argument_parser()
    parser.add_argument(
        "--exclude-class-names",
        default=None,
        help=(
            "Comma-separated class names to exclude from evaluation. Must "
            "match entries in the dataset metadata 'thing_classes'. Defaults "
            "to the WeDetect TCT_NGC negative list."
        ),
    )
    parser.add_argument(
        "--exclude-from-file",
        default=None,
        help=(
            "Path to a text file with one class name per line (lines starting "
            "with '#' are ignored). Mutually exclusive with "
            "--exclude-class-names."
        ),
    )
    return parser


def _resolve_exclude_list(args) -> List[str]:
    if args.exclude_class_names and args.exclude_from_file:
        raise SystemExit(
            "--exclude-class-names and --exclude-from-file are mutually exclusive."
        )
    if args.exclude_from_file:
        with open(args.exclude_from_file, "r") as f:
            names = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        return names
    if args.exclude_class_names:
        return [n.strip() for n in args.exclude_class_names.split(",") if n.strip()]
    return list(DEFAULT_NEGATIVE_CLASS_NAMES)


def main(args):
    logger = logging.getLogger("ovdino.eval_exclude_negative")
    exclude_names = _resolve_exclude_list(args)

    cfg = LazyConfig.load(args.config_file)
    # Swap the evaluator BEFORE applying ``--opts`` so user overrides on
    # ``dataloader.evaluator.*`` survive (mirrors WeDetect's ordering).
    _swap_evaluator(cfg, exclude_names, logger)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    default_setup(cfg, args)

    if cfg.train.fast_dev_run.enabled:
        cfg.train.max_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if not args.eval_only:
        raise SystemExit(
            "eval_exclude_negative.py only supports --eval-only mode; pass "
            "--eval-only and a checkpoint via train.init_checkpoint=..."
        )

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(
        cfg.train.init_checkpoint
    )
    if (
        cfg.train.model_ema.enabled
        and cfg.train.model_ema.use_ema_weights_for_eval_only
    ):
        ema.apply_model_ema(model)

    print(do_test(cfg, model, eval_only=True))


if __name__ == "__main__":
    args = _build_parser().parse_args()
    setup_dist_args(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
