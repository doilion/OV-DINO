"""Evaluators that drop one or more "negative" classes from COCO / LVIS evaluation.

Mirrors WeDetect's ``ExcludeClassCocoMetric`` (mmdet-side) but for detectron2 /
detrex evaluators used by OV-DINO. Two effects:

1. Predictions whose class label is in the excluded set are filtered out in
   ``process()`` before being serialized to the results JSON. They therefore
   cannot contribute false positives against the remaining classes.
2. The underlying COCOeval / LVISEval is restricted to the non-excluded
   category IDs, so the reported mAP / per-class table averages only over the
   classes that were not excluded.

Excluded classes are specified by **name** (matching ``thing_classes`` on the
dataset metadata) or by **contiguous index**. Names are recommended; they are
robust to dataset-id renumbering between splits.
"""

import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import COCOevalMaxDets
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detrex.data.evaluation.lvis_evaluation import (
    LVISFixedAPEvaluator,
    _evaluate_predictions_on_lvis,
)

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:  # pragma: no cover - matches detectron2 fallback
    COCOeval_opt = COCOeval


ExcludeSpec = Union[int, str, Iterable[Union[int, str]]]


def _resolve_excluded_contiguous_ids(
    exclude: ExcludeSpec,
    thing_classes: Sequence[str],
    logger: logging.Logger,
) -> List[int]:
    """Normalize the user-supplied exclusion spec into contiguous indices.

    A name that is not present in ``thing_classes`` is logged and skipped, so
    that a stale name (e.g. the WeDetect ``Urine-NILM`` case after a class
    merge) does not crash the run.
    """
    if isinstance(exclude, (int, str)):
        items: List[Union[int, str]] = [exclude]
    else:
        items = list(exclude)

    resolved: List[int] = []
    for item in items:
        if isinstance(item, str):
            try:
                resolved.append(thing_classes.index(item))
            except ValueError:
                logger.warning(
                    "ExcludeClassEvaluator: class name '%s' not in dataset "
                    "thing_classes (%d classes); skipping.",
                    item,
                    len(thing_classes),
                )
        else:
            idx = int(item)
            if 0 <= idx < len(thing_classes):
                resolved.append(idx)
            else:
                logger.warning(
                    "ExcludeClassEvaluator: contiguous id %d out of range "
                    "for %d-class dataset; skipping.",
                    idx,
                    len(thing_classes),
                )
    # de-duplicate while preserving order
    seen = set()
    unique: List[int] = []
    for idx in resolved:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


class ExcludeClassCOCOEvaluator(COCOEvaluator):
    """COCO evaluator that drops a configurable set of classes from scoring."""

    def __init__(
        self,
        dataset_name: str,
        exclude_class_names: Optional[Sequence[Union[int, str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name, *args, **kwargs)
        thing_classes = list(self._metadata.get("thing_classes") or [])
        if not thing_classes:
            raise ValueError(
                f"Dataset '{dataset_name}' has no 'thing_classes' metadata; "
                "ExcludeClassCOCOEvaluator needs class names to resolve "
                "exclusions."
            )
        self._thing_classes = thing_classes
        self._exclude_input = list(exclude_class_names or [])
        self._excluded_contiguous = _resolve_excluded_contiguous_ids(
            self._exclude_input, thing_classes, self._logger
        )
        self._excluded_names = [thing_classes[i] for i in self._excluded_contiguous]

        # Map contiguous id -> original dataset category id via metadata.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._contig2dataset = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
        else:
            self._contig2dataset = {i: i for i in range(len(thing_classes))}
        self._excluded_dataset_ids = [
            self._contig2dataset[i] for i in self._excluded_contiguous
        ]

        self._total_pred_instances = 0
        self._dropped_pred_instances = 0

        self._logger.info(
            "[ExcludeClassCOCOEvaluator] excluding %d class(es): "
            "names=%s contig_ids=%s dataset_ids=%s",
            len(self._excluded_contiguous),
            self._excluded_names,
            self._excluded_contiguous,
            self._excluded_dataset_ids,
        )

    # ---------- per-batch hook -------------------------------------------------
    def process(self, inputs, outputs):
        if not self._excluded_contiguous:
            return super().process(inputs, outputs)

        excluded = set(self._excluded_contiguous)
        filtered_outputs = []
        for output in outputs:
            new_output = dict(output)
            if "instances" in output:
                instances = output["instances"]
                pred_classes = instances.pred_classes
                if pred_classes.numel() > 0:
                    keep = ~np.isin(pred_classes.cpu().numpy(), list(excluded))
                    self._total_pred_instances += int(pred_classes.numel())
                    self._dropped_pred_instances += int((~keep).sum())
                    keep_tensor = torch.from_numpy(keep).to(pred_classes.device)
                    new_output["instances"] = instances[keep_tensor]
            filtered_outputs.append(new_output)

        super().process(inputs, filtered_outputs)

    # ---------- evaluation hook ------------------------------------------------
    def _eval_predictions(self, predictions, img_ids=None):
        """Re-implement just enough of d2's path to inject ``catIds`` filtering.

        Mirrors :meth:`COCOEvaluator._eval_predictions` line-for-line; the only
        custom step is constructing :class:`pycocotools.cocoeval.COCOeval` here
        (instead of delegating to ``_evaluate_predictions_on_coco``) so we can
        set ``params.catIds`` before ``evaluate()`` runs.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            num_classes = len(dataset_id_to_contiguous_id)
            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to %s", file_path)
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        excluded_dataset_ids = set(self._excluded_dataset_ids)
        kept_dataset_ids = [
            cid for cid in self._coco_api.getCatIds() if cid not in excluded_dataset_ids
        ]

        self._logger.info(
            "Evaluating predictions with %s COCO API on %d categories "
            "(%d excluded)...",
            "unofficial" if self._use_fast_impl else "official",
            len(kept_dataset_ids),
            len(excluded_dataset_ids),
        )

        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            if len(coco_results) == 0:
                self._results[task] = {
                    m: float("nan") for m in ("AP", "AP50", "AP75", "APs", "APm", "APl")
                }
                continue
            coco_eval = self._build_coco_eval(coco_results, task)
            if img_ids is not None:
                coco_eval.params.imgIds = img_ids
            coco_eval.params.catIds = kept_dataset_ids

            coco_eval.evaluate()
            coco_eval.accumulate()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                coco_eval.summarize()
            summary = buf.getvalue().strip()
            if summary:
                self._logger.info("\n%s", summary)

            kept_class_names = [
                name
                for idx, name in enumerate(self._thing_classes)
                if idx not in set(self._excluded_contiguous)
            ]
            res = self._derive_coco_results(
                coco_eval, task, class_names=kept_class_names
            )
            self._results[task] = res

        self._logger.info(
            "[ExcludeClassCOCOEvaluator] dropped %d / %d predicted instances "
            "from excluded classes.",
            self._dropped_pred_instances,
            self._total_pred_instances,
        )

    def _build_coco_eval(self, coco_results, iou_type):
        results = coco_results
        if iou_type == "segm":
            results = copy.deepcopy(coco_results)
            for c in results:
                c.pop("bbox", None)

        coco_dt = self._coco_api.loadRes(results)
        max_dets = self._max_dets_per_image
        use_max = max_dets is not None and max_dets[2] != 100
        if use_max:
            coco_eval = COCOevalMaxDets(self._coco_api, coco_dt, iou_type)
        else:
            coco_eval = (COCOeval_opt if self._use_fast_impl else COCOeval)(
                self._coco_api, coco_dt, iou_type
            )
        if iou_type != "keypoints":
            coco_eval.params.maxDets = max_dets or [1, 10, 100]
        return coco_eval


class ExcludeClassLVISFixedAPEvaluator(LVISFixedAPEvaluator):
    """LVIS fixed-AP evaluator with the same exclude-class behaviour."""

    def __init__(
        self,
        dataset_name: str,
        exclude_class_names: Optional[Sequence[Union[int, str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name, *args, **kwargs)
        thing_classes = list(self._metadata.get("thing_classes") or [])
        if not thing_classes:
            raise ValueError(
                f"Dataset '{dataset_name}' has no 'thing_classes' metadata; "
                "ExcludeClassLVISFixedAPEvaluator needs class names to "
                "resolve exclusions."
            )
        self._thing_classes = thing_classes
        self._excluded_contiguous = _resolve_excluded_contiguous_ids(
            list(exclude_class_names or []), thing_classes, self._logger
        )
        self._excluded_names = [thing_classes[i] for i in self._excluded_contiguous]

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._contig2dataset = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
        else:
            # LVIS path with no mapping: detrex remaps via +1 at serialization.
            self._contig2dataset = {i: i + 1 for i in range(len(thing_classes))}
        self._excluded_dataset_ids = [
            self._contig2dataset[i] for i in self._excluded_contiguous
        ]

        self._total_pred_instances = 0
        self._dropped_pred_instances = 0

        self._logger.info(
            "[ExcludeClassLVISFixedAPEvaluator] excluding %d class(es): "
            "names=%s contig_ids=%s dataset_ids=%s",
            len(self._excluded_contiguous),
            self._excluded_names,
            self._excluded_contiguous,
            self._excluded_dataset_ids,
        )

    def process(self, inputs, outputs):
        if not self._excluded_contiguous:
            return super().process(inputs, outputs)

        excluded = set(self._excluded_contiguous)
        filtered_outputs = []
        for output in outputs:
            new_output = dict(output)
            if "instances" in output:
                instances = output["instances"]
                pred_classes = instances.pred_classes
                if pred_classes.numel() > 0:
                    keep = ~np.isin(pred_classes.cpu().numpy(), list(excluded))
                    self._total_pred_instances += int(pred_classes.numel())
                    self._dropped_pred_instances += int((~keep).sum())
                    keep_tensor = torch.from_numpy(keep).to(pred_classes.device)
                    new_output["instances"] = instances[keep_tensor]
            filtered_outputs.append(new_output)

        super().process(inputs, filtered_outputs)

    def _eval_by_cat(self, by_cat):
        if self._excluded_contiguous:
            # ``by_cat`` is per-(contiguous-)category; drop the excluded ones
            # before LVIS API receives them.
            excluded_contig = set(self._excluded_contiguous)
            by_cat = [b for b in by_cat if b["category_id"] not in excluded_contig]
        super()._eval_by_cat(by_cat)
        self._restrict_lvis_results()

    def _eval_predictions(self, predictions):
        super()._eval_predictions(predictions)
        self._restrict_lvis_results()

    def _restrict_lvis_results(self):
        # The detrex LVIS path calls _evaluate_predictions_on_lvis which
        # constructs its own LVISEval internally. To keep the behavior simple
        # and aligned with WeDetect, we additionally re-run evaluation with
        # ``params.cat_ids`` restricted to non-excluded ids, and overwrite the
        # task result dict in-place.
        if not self._excluded_contiguous:
            return
        excluded_dataset_ids = set(self._excluded_dataset_ids)
        kept_cat_ids = [
            cid for cid in self._lvis_api.get_cat_ids() if cid not in excluded_dataset_ids
        ]
        # Load the predictions we just dumped. Require an explicit output_dir
        # so we never accidentally pick up a stale json from cwd.
        if not self._output_dir:
            self._logger.warning(
                "[ExcludeClassLVISFixedAPEvaluator] output_dir is None; "
                "cannot re-evaluate with restricted cat_ids. Pass "
                "dataloader.evaluator.output_dir=... to enable it."
            )
            return
        result_json = os.path.join(self._output_dir, "lvis_instances_results.json")
        if not os.path.exists(result_json):
            self._logger.warning(
                "[ExcludeClassLVISFixedAPEvaluator] expected results json at "
                "%s but did not find it; AP table reflects the full "
                "category list.",
                result_json,
            )
            return
        with open(result_json, "r") as f:
            lvis_results = json.load(f)
        lvis_results = [r for r in lvis_results if r["category_id"] not in excluded_dataset_ids]
        from lvis import LVISEval, LVISResults

        max_dets = self._max_dets_per_image if self._max_dets_per_image > 0 else 300
        for task in list(self._results.keys()):
            if task == "box_proposals":
                continue
            res = LVISResults(self._lvis_api, lvis_results, max_dets=max_dets)
            lvis_eval = LVISEval(self._lvis_api, res, task)
            lvis_eval.params.cat_ids = kept_cat_ids
            lvis_eval.params.max_dets = max_dets
            lvis_eval.run()
            lvis_eval.print_results()
            metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"]
            raw = lvis_eval.get_results()
            new_res = OrderedDict(
                (m, float(raw[m] * 100)) for m in metrics if m in raw
            )
            self._logger.info(
                "Excluded-class LVIS results for %s:\n%s",
                task,
                create_small_table(new_res),
            )
            self._results[task] = new_res

        self._logger.info(
            "[ExcludeClassLVISFixedAPEvaluator] dropped %d / %d predicted "
            "instances from excluded classes.",
            self._dropped_pred_instances,
            self._total_pred_instances,
        )
