#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# Most code in this file is copied from detectron2/tools/deploy/export_model.py
import argparse
import itertools
import os
from typing import Dict, List, Tuple

import detectron2.data.transforms as T
import numpy as np
import onnxruntime as ort
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import detection_utils
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detrex.data.datasets import clean_words_or_phrase, template_meta
from torch import Tensor, nn

_category_names = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]


def setup_cfg(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def export_caffe2_tracing(cfg, torch_model, inputs):
    from detectron2.export import Caffe2Tracer

    tracer = Caffe2Tracer(cfg, torch_model, inputs)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=inputs)
        return caffe2_model
    elif args.format == "onnx":
        try:
            import onnx
        except:
            import torch.onnx as onnx

        onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, os.path.join(args.output, "model.onnx"))
    elif args.format == "torchscript":
        ts_model = tracer.export_torchscript()
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)


# experimental. API not yet final
def export_scripting(torch_model):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    assert args.format == "torchscript", "Scripting only supports torchscript format."

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(
                self, inputs: Tuple[Dict[str, torch.Tensor]]
            ) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(
                self, inputs: Tuple[Dict[str, torch.Tensor]]
            ) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, args.output)
    # TODO inference in Python now missing postprocessing glue code
    return None


def tokenize_category_names(torch_model, category_names):
    category_names = [
        [
            [
                template.format(clean_words_or_phrase(cat_name))
                for template in template_meta[torch_model.inference_template]
            ]
            for cat_name in batch_cat_names
        ]
        for batch_cat_names in category_names
    ]
    category_names = list(itertools.chain(*category_names))
    if isinstance(category_names[0], list):
        # use the following code to get input_ids
        # input_ids = torch.stack(
        #     [
        #         torch_model.language_backbone.tokenizer(name, return_mask=True)[
        #             "input_ids"
        #         ].squeeze(0)
        #         for name in category_names
        #     ],
        #     dim=0,
        # )

        # or use the following code to get text_embed
        with torch.no_grad():
            text_embed = torch.stack(
                [torch_model.language_backbone(name) for name in category_names], dim=0
            )  # [bs*num_classes, num_templates, embed_dim]
            text_embed = text_embed.mean(1).cpu()

    return text_embed


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert (
        TORCH_VERSION >= (1, 8) and len(inputs) == 1
    ), "Tracing expects one input image!"
    image = inputs[0]["image"]
    height = inputs[0]["height"]
    width = inputs[0]["width"]
    category_names = inputs[0]["category_names"]  # tensor, [bs*num_classes, embed_dim]
    inputs = [
        {
            "category_names": category_names,
            "height": torch.tensor(height),
            "image": image,
            "width": torch.tensor(width),
        }
    ]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(
            traceable_model,
            (category_names, height, image, width),
        )
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(
                traceable_model,
                (category_names, height, image, width),
                f,
                input_names=["category_names", "height", "image", "width"],
                output_names=["instances"],
                dynamic_axes={
                    "category_names": {0: "num_classes"},
                    "image": {1: "height", 2: "width"},
                },
                opset_version=13,
                verbose=True,
                do_constant_folding=False,  # set constant folding to False will speed the export
            )
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(
            ts_model(
                input["category_names"], input["height"], input["image"], input["width"]
            )
        )[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs(args, torch_model):

    if args.sample_image is None:
        # get a first batch from dataset
        data_loader = instantiate(cfg.dataloader.test)
        first_batch = next(iter(data_loader))
        assert "category_names" in first_batch[0]
        image = first_batch[0]["image"]
        category_names = [first_batch[0]["category_names"]]
        height = first_batch[0]["height"]
        width = first_batch[0]["width"]
        input_ids = tokenize_category_names(torch_model, category_names)

        first_batch[0]["image"] = image
        first_batch[0]["category_names"] = input_ids
        first_batch[0]["height"] = torch.tensor(height)
        first_batch[0]["width"] = torch.tensor(width)
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(args.sample_image, format="RGB")
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # Use default COCO categories if sample_categories not provided
        category_names = [
            (
                args.sample_categories
                if args.sample_categories is not None
                else _category_names
            )
        ]
        print("category_names: ", category_names)
        input_ids = tokenize_category_names(torch_model, category_names)

        inputs = {
            "category_names": input_ids,  # Fix typo in key name
            "image": image,
            "height": torch.tensor(height),
            "width": torch.tensor(width),
        }

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


def main() -> None:
    global logger, cfg, args
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--sample-image", default=None, type=str, help="sample image for input"
    )
    parser.add_argument(
        "--sample-categories",
        default=None,
        nargs="+",
        type=str,
        help="sample categories for input",
    )
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--compare-export", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = instantiate(cfg.model)
    torch_model.to(cfg.train.device)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.train.init_checkpoint)
    torch_model.eval()

    # convert and save model
    if args.export_method == "caffe2_tracing":
        sample_inputs = get_sample_inputs(args, torch_model)
        exported_model = export_caffe2_tracing(cfg, torch_model, sample_inputs)
    elif args.export_method == "scripting":
        exported_model = export_scripting(torch_model)
    elif args.export_method == "tracing":
        sample_inputs = get_sample_inputs(args, torch_model)
        exported_model = export_tracing(torch_model, sample_inputs)

    if args.compare_export:
        logger.info("Running comparison between original model and exported model ...")
        with torch.no_grad():
            torch_model.eval()
            torch_output = torch_model(
                sample_inputs
            )  # pred_boxes, scores, pred_classes

        # onnx model output
        if args.format == "onnx":
            onnx_model_path = os.path.join(args.output, "model.onnx")
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 4
            session_options.intra_op_num_threads = 4
            ort_session = ort.InferenceSession(
                onnx_model_path,
                sess_options=session_options,
                provider=[
                    "CPUExecutionProvider",
                ],
            )

            # Prepare inputs for ONNX runtime
            ort_inputs = {
                "category_names": sample_inputs[0]["category_names"].cpu().numpy(),
                "height": sample_inputs[0]["height"].cpu().numpy(),
                "image": sample_inputs[0]["image"].cpu().numpy(),
                "width": sample_inputs[0]["width"].cpu().numpy(),
            }

            ort_outputs = ort_session.run(
                None, ort_inputs
            )  # pred_boxes, pred_classes, scores, (height, width)

            # Compare outputs
            # NOTE: you need to set breakpoint here to compare the outputs
            logger.info("Comparing outputs...")
            # for torch_out, ort_out in zip(torch_output, ort_outputs):
            #     diff = np.abs(torch_out.cpu().numpy() - ort_out).max()
            #     logger.info(f"Max difference: {diff}")

    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info(
            "Running evaluation ... this takes a long time if you export to CPU."
        )
        data_loader = instantiate(cfg.dataloader.test)
        evaluator = instantiate(cfg.dataloader.evaluator)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Success.")


if __name__ == "__main__":
    main()  # pragma: no cover
