# AGENTS.md

This file provides repository-specific guidance to Codex when working in this project.

## Primary Role In This Repo

Codex is primarily used as a reviewer and debugging partner for OV-DINO changes.

- Default to review-first behavior when asked to "review" a change.
- Prioritize correctness and regression risk over style suggestions.
- Be explicit about what was verified versus what is inferred.
- Do not claim reproduction or metric improvements without concrete evidence.

## Review Priorities

When reviewing changes, focus on these areas first:

1. Training and inference correctness
2. Checkpoint compatibility
3. DDP and AMP safety
4. Config regressions and dataset assumptions
5. Evaluation protocol and metric changes

Use [`.github/REVIEW_CHECKLIST.md`](/root/code/OV-DINO/.github/REVIEW_CHECKLIST.md) as the default checklist.

## Repository Map

Most important paths:

- `ovdino/projects/ovdino/modeling`
- `ovdino/projects/ovdino/configs`
- `ovdino/tools/train_net.py`
- `ovdino/demo`
- `ovdino/configs/common`

## Project Constraints

- This repository uses Detectron2 LazyConfig and detrex conventions.
- Datasets live under `datas/` at the repository root.
- Checkpoint and config compatibility matter more than aggressive cleanup.
- Training outputs belong under `wkdrs/`; do not commit generated artifacts, datasets, or caches.
- Large benchmark claims should include the exact config, checkpoint, and evaluation split.

## Review Workflow

- For local review packages, prefer `bash scripts/gen_review.sh`.
- For GitHub review, expect PRs to follow [`.github/pull_request_template.md`](/root/code/OV-DINO/.github/pull_request_template.md).
- If a change touches model logic, matching, losses, evaluators, or config defaults, call that out explicitly.
- If tests or training runs were not executed, say so clearly.

