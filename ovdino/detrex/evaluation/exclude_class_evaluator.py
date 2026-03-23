import copy
import itertools
import json
import os

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import COCOevalMaxDets
from detectron2.utils.file_io import PathManager
from pycocotools.cocoeval import COCOeval

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class ExcludeClassCOCOEvaluator(COCOEvaluator):
    """COCOEvaluator that excludes specific categories from AP computation.

    Inherits from Detectron2's COCOEvaluator and overrides _eval_predictions()
    to filter out excluded categories at the pycocotools level by setting
    coco_eval.params.catIds before evaluation.

    Args:
        dataset_name: Name of the dataset to evaluate.
        exclude_class_names: List of category names to exclude from evaluation.
        **kwargs: All other arguments passed to COCOEvaluator.
    """

    def __init__(self, dataset_name, exclude_class_names=None, **kwargs):
        super().__init__(dataset_name, **kwargs)
        self._exclude_class_names = exclude_class_names or []
        self._exclude_cat_ids = set()
        self._exclude_contiguous_ids = set()

        if self._exclude_class_names:
            id_map = self._metadata.thing_dataset_id_to_contiguous_id
            class_names = self._metadata.thing_classes
            reverse_map = {v: k for k, v in id_map.items()}
            for name in self._exclude_class_names:
                if name in class_names:
                    contiguous_id = class_names.index(name)
                    coco_id = reverse_map[contiguous_id]
                    self._exclude_cat_ids.add(coco_id)
                    self._exclude_contiguous_ids.add(contiguous_id)
                    self._logger.info(
                        f"Excluding category: '{name}' "
                        f"(contiguous_id={contiguous_id}, coco_id={coco_id})"
                    )
                else:
                    self._logger.warning(
                        f"Exclude category '{name}' not found in dataset classes"
                    )

    def _eval_predictions(self, predictions, img_ids=None):
        """Evaluate predictions with excluded categories filtered out."""
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # Unmap category ids (contiguous -> COCO original)
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        # Filter out excluded category predictions
        total_before = len(coco_results)
        coco_results = [
            r for r in coco_results if r["category_id"] not in self._exclude_cat_ids
        ]
        total_after = len(coco_results)
        self._logger.info(
            f"Filtered predictions: {total_before} -> {total_after} "
            f"(removed {total_before - total_after} from excluded categories)"
        )

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        # Build filtered category list and class names
        all_cat_ids = sorted(self._coco_api.getCatIds())
        filtered_cat_ids = [
            cid for cid in all_cat_ids if cid not in self._exclude_cat_ids
        ]

        all_class_names = self._metadata.get("thing_classes")
        if all_class_names:
            filtered_class_names = [
                name for name in all_class_names
                if name not in self._exclude_class_names
            ]
        else:
            filtered_class_names = None

        self._logger.info(
            f"Evaluating {len(filtered_cat_ids)} categories "
            f"(excluded {len(self._exclude_cat_ids)})"
        )

        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                self._evaluate_with_exclusion(
                    coco_results,
                    task,
                    filtered_cat_ids,
                    img_ids=img_ids,
                )
                if len(coco_results) > 0
                else None
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=filtered_class_names
            )
            self._results[task] = res

    def _evaluate_with_exclusion(
        self, coco_results, iou_type, filtered_cat_ids, img_ids=None
    ):
        """Run COCOeval with filtered category IDs."""
        assert len(coco_results) > 0

        if iou_type == "segm":
            coco_results = copy.deepcopy(coco_results)
            for c in coco_results:
                c.pop("bbox", None)

        coco_dt = self._coco_api.loadRes(coco_results)

        # Match upstream COCOEvaluator logic: use COCOevalMaxDets when
        # max_dets_per_image has a non-default third element (!= 100),
        # so that summarize() reports AP/AR at the custom maxDets value.
        max_dets = self._max_dets_per_image
        if max_dets is not None and len(max_dets) >= 3 and max_dets[2] != 100:
            coco_eval = COCOevalMaxDets(self._coco_api, coco_dt, iou_type)
        else:
            coco_eval = (COCOeval_opt if self._use_fast_impl else COCOeval)(
                self._coco_api, coco_dt, iou_type
            )

        # Set filtered category IDs — the key exclusion mechanism
        coco_eval.params.catIds = filtered_cat_ids

        if iou_type != "keypoints":
            coco_eval.params.maxDets = max_dets if max_dets is not None else [1, 10, 100]

        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval
