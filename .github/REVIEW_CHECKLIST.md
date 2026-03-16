# Code Review Checklist for OV-DINO

Use this checklist when reviewing code changes. Paste it alongside a diff when asking Codex to review.

## Critical (must check)

### Training Correctness
- [ ] Loss computation unchanged or change is mathematically justified
- [ ] Gradient flow not broken (no detach() on tensors that need gradients)
- [ ] Label assignment logic (Hungarian matching) untouched or change is validated
- [ ] Denoising training (DN groups, noise scale) consistent with config
- [ ] Learning rate schedule and optimizer config correct

### Inference Correctness
- [ ] Pre-trained checkpoint loads without errors or missing/unexpected keys
- [ ] NMS / post-processing thresholds unchanged (or change is documented)
- [ ] Category mapping and text embeddings match expected format
- [ ] Output format (boxes, scores, labels) unchanged

### DDP & AMP Safety
- [ ] No operations that break gradient synchronization across ranks
- [ ] All-reduce / broadcast calls balanced across processes
- [ ] No float16 accumulation in loss terms (use float32 for loss)
- [ ] `autocast` scopes correct — no casting where it shouldn't happen
- [ ] Batch norm / sync batch norm handled correctly in DDP

### Checkpoint Compatibility
- [ ] State dict keys unchanged (or migration code provided)
- [ ] New parameters have proper default initialization
- [ ] Removed parameters handled gracefully on load (strict=False with logging)

## Important (should check)

### Config Regression
- [ ] Default values in configs unchanged (or change is intentional)
- [ ] New config fields have sensible defaults
- [ ] Config inheritance chain (`_base_`) not broken
- [ ] Environment variable usage consistent (`DETECTRON2_DATASETS`, `MODEL_ROOT`, etc.)

### Data Pipeline
- [ ] Dataset registration IDs unchanged
- [ ] Annotation format assumptions (COCO JSON) preserved
- [ ] Data augmentation pipeline order and parameters correct
- [ ] Dataloader num_workers, pin_memory, batch_size sensible

### Evaluation Metrics
- [ ] Evaluation protocol matches paper / baseline
- [ ] AP computation (COCO / LVIS) uses correct evaluator
- [ ] Category filtering for zero-shot / open-vocabulary correct
- [ ] Before/after numbers provided if metrics are expected to change

## Nice to have

### Code Quality
- [ ] No dead code or commented-out blocks left behind
- [ ] Variable names clear and consistent with codebase conventions
- [ ] Black / isort / flake8 formatting followed (line length 100)
- [ ] No hard-coded absolute paths

### Resource Usage
- [ ] Memory usage acceptable (no unnecessary tensor copies)
- [ ] No CPU-GPU transfers in hot loops
- [ ] CUDA kernel launches reasonable
- [ ] Logging frequency not excessive

---

## Codex Review Prompt Template

Paste this along with the diff output from `scripts/gen_review.sh`:

```
Review this diff for OV-DINO (open-vocabulary object detection, Swin + BERT + DINO architecture).
Please focus on:
- Training/inference correctness
- AMP/DDP safety
- Checkpoint loading compatibility
- Config regressions
- Evaluation metric changes
Use the checklist below to structure your review.
```
