# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo is the official implementation of **OV-DINO** (open-vocabulary detection). The four top-level directories that the shell scripts hard-code (relative to the repo root) are:

- [ovdino/](ovdino/) — all source code (vendored `detectron2-717ab9/`, vendored `detrex/`, the `projects/ovdino/` model, `configs/`, `tools/`, `demo/`, `scripts/`).
- [datas/](datas/) — datasets (`coco/`, `lvis/`, `o365/`, `custom/`). Exported to `DETECTRON2_DATASETS`.
- [inits/](inits/) — pretrained checkpoints. `inits/huggingface/` is used as `HF_HOME` for offline BERT loading; `inits/ovdino/` holds OV-DINO weights; `inits/swin/` and `inits/sam2/` hold their respective backbones. Exported to `MODEL_ROOT`.
- [wkdrs/](wkdrs/) — every script writes outputs (checkpoints, logs, eval dirs) under here.

The shell scripts derive `$root_dir` via `realpath $(dirname $0)/../../`, so they must be invoked from `ovdino/scripts/...` (or with that relative path preserved) for `DETECTRON2_DATASETS`, `MODEL_ROOT`, and `HF_HOME` to resolve correctly.

## Install

Two install steps build native code; both are required after a clean clone:

```bash
cd ovdino
python -m pip install -e detectron2-717ab9   # vendored detectron2 fork
pip install -e ./                            # builds detrex + the CUDA op detrex._C
```

`setup.py` compiles `detrex/layers/csrc/*.cu` into `detrex._C`. The default conda env (`ovdino`) is pinned to PyTorch 1.13.1 + CUDA 11.6; the optional `ovsam` env (for OV-SAM = OV-DINO + SAM2) uses PyTorch 2.3.1 + CUDA 12.1. Set `CUDA_HOME` before pip install if the system CUDA differs.

## Common Commands

All training/eval/inference flows are thin wrappers that ultimately call `ovdino/tools/train_net.py` with a detrex LazyConfig file. Run scripts from the repo root:

```bash
# Zero-shot eval (config decides dataset: COCO / LVIS-MiniVal / LVIS-Val)
bash ovdino/scripts/eval.sh <config.py> <ckpt.pth> <output_dir>

# Fine-tune (output dir is wkdrs/<config_name>)
bash ovdino/scripts/finetune.sh <config.py> <init_ckpt.pth>

# Pretrain (single-node defaults; multi-node via NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT env vars)
bash ovdino/scripts/pretrain.sh <config.py>

# Inference demo: category_names is a space-separated string; multi-word classes use underscores
bash ovdino/scripts/demo.sh <demo_config.py> <ckpt.pth> "class0 class1 ..." <input> <output>

# Gradio web demo on :7860 (OV-SAM if running in ovsam env)
bash ovdino/scripts/app.sh <demo_config.py> <ckpt.pth>
```

Configs live in [ovdino/projects/ovdino/configs/](ovdino/projects/ovdino/configs/) — naming pattern `ovdino_<backbone>_<text_enc>_<task>_<dataset>[_<sched>].py`. Eval/fine-tune configs already point at the matching dataset; do not pair an LVIS config with COCO data.

LazyConfig overrides use dotted `key=value` on the command line, e.g. `train.init_checkpoint=...` `dataloader.evaluator.output_dir=...` `model.num_classes=...`. The scripts append these after `--opts` for demos and directly for `train_net.py`.

There is also [ovdino/tools/hydra_train_net.py](ovdino/tools/hydra_train_net.py) for slurm + hydra launches (`configs/hydra/slurm/<cluster>.yaml`), [tools/benchmark.py](ovdino/tools/benchmark.py) for train/eval/data speed, and [tools/visualize_data.py](ovdino/tools/visualize_data.py) / [tools/visualize_json_results.py](ovdino/tools/visualize_json_results.py) for sanity checks.

## Architecture

**Three stacked codebases.** The repo vendors two upstream libraries plus its own model:

1. `ovdino/detectron2-717ab9/` — pinned detectron2 fork. Installed editable; do not assume system detectron2.
2. `ovdino/detrex/` — IDEA's DETR-family library, installed as the editable package `detrex` (version 0.3.0). Provides the backbone/neck/transformer/criterion/matcher/EMA building blocks and the LazyConfig-driven dataset registration (`detrex/data/datasets/register_*_ovd.py`).
3. `ovdino/projects/ovdino/` — the OV-DINO model itself: [modeling/ovdino.py](ovdino/projects/ovdino/modeling/ovdino.py) (`OVDINO` nn.Module = vision backbone + language backbone + neck + DINO transformer + criterion), [modeling/dino_transformer.py](ovdino/projects/ovdino/modeling/dino_transformer.py), [modeling/dn_criterion.py](ovdino/projects/ovdino/modeling/dn_criterion.py), [modeling/two_stage_criterion.py](ovdino/projects/ovdino/modeling/two_stage_criterion.py).

**Config-as-code (detrex/detectron2 LazyConfig).** Each top-level config in `projects/ovdino/configs/` imports a base model from `configs/models/`, a dataloader from `configs/common/data/<dataset>_ovd.py`, a schedule from `configs/common/{coco,common,pretrain}_schedule.py`, and the training defaults from [configs/common/train.py](ovdino/configs/common/train.py). Components are constructed by `detectron2.config.instantiate` at runtime, which is why CLI overrides target dotted attribute paths.

**Open-vocabulary dataset registration.** Each dataset has a paired `<name>_ovd.py` (builds the dataloader/evaluator) and `register_<name>_ovd.py` (registers a detectron2 dataset that emits language prompts alongside images). The "OVD" naming distinguishes them from stock detectron2 datasets — always use the `_ovd` variants for OV-DINO.

**Custom datasets.** Follow inline instructions in [ovdino/configs/common/data/custom_ovd.py](ovdino/configs/common/data/custom_ovd.py): convert to COCO format, edit the meta-info block to list your class names, then use `ovdino_swin_tiny224_bert_base_ft_custom_24ep.py` as the fine-tune config.

**Language backbone offline.** Scripts set `TRANSFORMERS_OFFLINE=1` and `HF_HOME=$root_dir/inits/huggingface`, so the BERT text encoder must already be cached under `inits/huggingface/` before any training/eval — first-time setup needs to populate that cache while online.
