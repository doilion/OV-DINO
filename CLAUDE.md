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

- Use [`.github/pull_request_template.md`](.github/pull_request_template.md) for every PR.
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

## Testing

There are no project-owned tests. Detectron2's vendored tests exist in `detectron2-717ab9/tests/` but are not part of the OV-DINO workflow. Validation is done by running eval scripts against benchmarks.

## Common Commands

All commands run from `ovdino/` directory. Set environment first:
```bash
export root_dir=$(realpath ./OV-DINO)  # or parent of ovdino/
cd $root_dir/ovdino
```

### Installation
```bash
python -m pip install -e detectron2-717ab9
pip install -e ./  # compiles CUDA extensions in detrex/layers/csrc — requires CUDA_HOME
```

### Evaluation
```bash
bash scripts/eval.sh <config_file> <checkpoint> <output_dir>
# Example: COCO zero-shot
bash scripts/eval.sh projects/ovdino/configs/ovdino_swin_tiny224_bert_base_eval_coco.py ../inits/ovdino/<ckpt>.pth ../wkdrs/eval_ovdino
```

### Fine-tuning
```bash
# Note: checkpoint is $2; output_dir is auto-derived from config name into wkdrs/
# MODEL_ROOT env var (default: $root_dir/inits/) is used for backbone/init weights
bash scripts/finetune.sh <config_file> <pretrained_checkpoint>
```

### Pre-training (multi-node)
```bash
# Output dir auto-derived; MODEL_ROOT used for init weights
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
- `detrex/modeling/language_backbone/bert.py` — BERT encoder wrapper (known quirks: projection param is named `text_porj` [typo]; `pooling_mode="max"` actually takes the [EOS] token, not a max-pool; `post_tokenize=True` calls `.cuda()` directly inside forward)
- `detectron2-717ab9/` — Detectron2 submodule (detection framework dependency)
- `configs/common/` — Shared configs for data loaders, training, optimization, schedules
- `tools/train_net.py` — Main training/eval entry point
- `demo/` — `demo.py` (CLI), `app.py` (Gradio web UI), `predictors.py` (inference class)
- `scripts/` — Shell wrappers: `eval.sh`, `finetune.sh`, `pretrain.sh`, `demo.sh`, `app.sh`

### Configuration System
Uses Detectron2's **LazyConfig** with `LazyCall` (`L(...)`) — Python files, not YAML. Model configs in `projects/ovdino/configs/models/` compose the full model inline. Task configs (eval, finetune, pretrain) import a model config and override common configs from `configs/common/`.

### Training Internals (`projects/ovdino/train_net.py`)
- **Hardcoded per-group learning rates**: backbone and `reference_points`/`sampling_offsets` get 10x lower LR (2e-5 vs 2e-4). These are not read from config — they are hardwired in `do_train`.
- AMP and gradient clipping are merged into a single `Trainer` class.
- `num_classes` (training, e.g. 150) and `test_num_classes` (eval, e.g. 80) are separate model params — the split is important for zero-shot eval on datasets with different class counts.

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

## In-Progress Work: BioMistral Integration on TCT_NGC

### Overview
Replace BERT text encoder with BioMistral-7B precomputed embeddings for medical cell detection (31 classes: 20 base + 11 novel). Uses adapter MLP (4096→768) + STEGO correspondence distillation loss.

### Key Files
- `detrex/layers/biomistral_adapter.py` — BioMistralAdapterMLP (LayerNorm→Linear→GELU→Dropout→Linear)
- `detrex/modeling/language_backbone/precomputed_embedding.py` — PrecomputedEmbeddingBackbone (drop-in for BERTEncoder)
- `projects/ovdino/modeling/correspondence_loss.py` — CorrespondenceDistillationLoss (STEGO)
- `projects/ovdino/modeling/ovdino.py` — adapter_mlp, correspondence_loss, freeze_visual params
- `projects/ovdino/configs/models/ovdino_swin_tiny224_biomistral.py` — Model config
- `projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase1_tct_ngc.py` — Phase 1 config
- `projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase2_tct_ngc.py` — Phase 2 config
- `embeddings/biomistral_tct_ngc.pt` — Precomputed 31×4096 embeddings (in repo)
- `embeddings/adapter_prealigned.pth` — Pre-aligned adapter weights, Spearman=0.866 (in repo)
- `scripts/extract_biomistral_embeddings.py` — Phase 0a: extract embeddings
- `scripts/prealign_adapter.py` — Phase 0b: STEGO pre-alignment

### Completed Steps
- [x] Phase 0a: BioMistral embedding extraction (31 classes, 4096d, mean pooling, L2 normalized)
- [x] Phase 0b: Adapter pre-alignment (negative_pressure=0.6, Spearman=0.866)
- [x] Code integration: adapter MLP, PrecomputedEmbeddingBackbone, CorrespondenceDistillationLoss
- [x] Config fixes: checkpoint path, test_num_classes=31, negative_pressure 0.4→0.6
- [x] Hyphen normalization fix in PrecomputedEmbeddingBackbone (high-grade vs high grade)
- [x] Smoke test passed: loss_corr outputs correctly, no OOM on 8×2080Ti, all 20 classes matched

### Next Steps
- [ ] Phase 1: Freeze visual, train adapter+ClassEmbed+BBoxEmbed (8 epochs)
- [ ] Phase 1 eval: base + novel AP
- [ ] Phase 2: Unfreeze all, joint fine-tuning (16 epochs, backbone 0.1x LR)
- [ ] Phase 2 eval: base + novel AP

### Training Commands
```bash
# Phase 1 (8 epochs, freeze visual)
bash scripts/finetune.sh \
  projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase1_tct_ngc.py \
  ../inits/ovdino/ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth

# Phase 2 (16 epochs, full fine-tuning)
bash scripts/finetune.sh \
  projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase2_tct_ngc.py \
  ./wkdrs/ovdino_swin_tiny224_biomistral_phase1_tct_ngc/model_final.pth

# Eval (base / novel)
bash scripts/eval.sh projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_base.py <ckpt> <output>
bash scripts/eval.sh projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_novel.py <ckpt> <output>
```

### BERT Baseline Reference (TCT_NGC)
- Base mAP: ~26-33 | Novel mAP: ~8.81

### Design Decisions
- `negative_pressure=0.6` (not 0.4) — teacher sim min=0.44, b=0.4 gives 0% repulsive pairs → collapse
- Correspondence loss weight=100.0 is internal, NOT in criterion weight_dict (avoids double-apply)
- `tools/train_net.py` is the real entry point (uses cfg.optimizer); `projects/ovdino/train_net.py` is deprecated dead code
- Reference implementation: https://github.com/doilion/YOLO-WORLD-MEDICAL

## Important Notes
- LVIS Val evaluation requires ~250GB RAM
- Default O365 pre-training uses batch size 64 on 2 nodes x 8 A100 GPUs
- Pre-trained checkpoints go in `inits/ovdino/`; Swin backbone in `inits/swin/`
- Training outputs go to `wkdrs/<config_name>/`
- Distributed training uses PyTorch DDP with optional AMP and gradient clipping (max_norm=0.1)
- Keep datasets, model weights, logs, and other generated artifacts out of PRs unless explicitly requested
