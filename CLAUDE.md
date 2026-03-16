# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OV-DINO is a unified open-vocabulary object detection model combining a Swin Transformer visual backbone, BERT language backbone, and DINO transformer detection architecture with Language-Aware Selective Fusion. It achieves state-of-the-art zero-shot detection on COCO and LVIS benchmarks.

## Collaboration Protocol

Claude is the implementation agent for this repository.

- Make code changes, documentation updates, and PR-ready patches.
- Do not merge directly to the default branch.
- Prefer focused PR-sized changes over broad refactors.
- Preserve checkpoint compatibility and config defaults unless the task explicitly requires changing them.
- If a change touches training, evaluation, model outputs, data assumptions, or checkpoint loading, summarize the risk in the PR description.
- If validation is partial, state exactly what was run and what remains unverified.
- If a result is inferred from code inspection rather than measured, label it as an inference.

## PR Expectations

- Use [`.github/pull_request_template.md`](/root/code/OV-DINO/.github/pull_request_template.md) for every PR.
- When changing model logic, configs, evaluators, or checkpoint loading, include a short reviewer guide.
- Ask Codex to review the PR after implementation. Recommended PR comment:

```text
@codex review
Please focus on:
- training/inference correctness
- AMP/DDP safety
- checkpoint loading compatibility
- config regressions
- evaluation metric changes
```

- For local review requests, generate a package with `bash scripts/gen_review.sh`.

## Common Commands

All commands run from `ovdino/` directory. Set environment first:
```bash
export root_dir=$(realpath ./OV-DINO)  # or parent of ovdino/
cd $root_dir/ovdino
```

### Installation
```bash
python -m pip install -e detectron2-717ab9
pip install -e ./
```

### Evaluation
```bash
bash scripts/eval.sh <config_file> <checkpoint> <output_dir>
# Example: COCO zero-shot
bash scripts/eval.sh projects/ovdino/configs/ovdino_swin_tiny224_bert_base_eval_coco.py ../inits/ovdino/<ckpt>.pth ../wkdrs/eval_ovdino
```

### Fine-tuning
```bash
bash scripts/finetune.sh <config_file> <pretrained_checkpoint>
```

### Pre-training (multi-node)
```bash
NNODES=2 NODE_RANK=0 MASTER_PORT=$PORT MASTER_ADDR=$ADDR bash scripts/pretrain.sh <config_file>
```

### Demo Inference
```bash
# CLI: category names are space-separated, multi-word classes use underscores
bash scripts/demo.sh <config> <checkpoint> "cat dog person" <input_images> <output_dir>
# Web UI (Gradio at http://127.0.0.1:7860)
bash scripts/app.sh <config> <checkpoint>
```

## Architecture

### Key Directories (under `ovdino/`)
- `projects/ovdino/modeling/` — Core model: `ovdino.py` (main class), `dino_transformer.py` (encoder-decoder), `dn_criterion.py` (denoising loss)
- `projects/ovdino/configs/` — Model and task configs (eval, finetune, pretrain, demo variants)
- `detrex/` — Reusable detection transformer framework (modeling, layers with CUDA extensions, data, checkpoint utils)
- `detrex/modeling/language_backbone/bert.py` — BERT encoder wrapper
- `detectron2-717ab9/` — Detectron2 submodule (detection framework dependency)
- `configs/common/` — Shared configs for data loaders, training, optimization, schedules
- `tools/train_net.py` — Main training/eval entry point
- `demo/` — `demo.py` (CLI), `app.py` (Gradio web UI), `predictors.py` (inference class)
- `scripts/` — Shell wrappers: `eval.sh`, `finetune.sh`, `pretrain.sh`, `demo.sh`, `app.sh`

### Configuration System
Uses Detectron2's **LazyConfig** (Python files, not YAML). Model configs in `projects/ovdino/configs/` import and override common configs from `configs/common/`. Training entry point is `tools/train_net.py` with custom trainer in `projects/ovdino/train_net.py` supporting AMP and multi-level learning rates.

### Data Pipeline
- Datasets expected in `datas/` at repo root: `coco/`, `lvis/` (symlinked to COCO images), `o365/`, `custom/`
- All use COCO JSON annotation format
- Custom datasets must follow the spec in `configs/common/data/custom_ovd.py`
- Environment variable `DETECTRON2_DATASETS` controls dataset root

### Environment Variables
- `DETECTRON2_DATASETS` — Dataset root (default: `$root_dir/datas/`)
- `MODEL_ROOT` — Checkpoint root (default: `$root_dir/inits/`)
- `HF_HOME` — HuggingFace cache (default: `$root_dir/inits/huggingface`)
- `CUDA_HOME` — Required if not using default CUDA 11.6

## Code Style
- Line length: 100
- Formatting: Black 22.3.0, isort 4.3.21, Flake8 3.8.1
- Type checking: mypy (Python 3.7 target)
- isort sections: FUTURE, STDLIB, THIRDPARTY, detrex (myself), FIRSTPARTY, LOCALFOLDER

## Important Notes
- LVIS Val evaluation requires ~250GB RAM
- Default O365 pre-training uses batch size 64 on 2 nodes x 8 A100 GPUs
- Pre-trained checkpoints go in `inits/ovdino/`; Swin backbone in `inits/swin/`
- Training outputs go to `wkdrs/<config_name>/`
- Distributed training uses PyTorch DDP with optional AMP and gradient clipping (max_norm=0.1)
- Keep datasets, model weights, logs, and other generated artifacts out of PRs unless explicitly requested
