## Summary

<!-- What problem does this PR solve? Keep it short and concrete. -->

## What Changed

<!-- List the main implementation changes. -->

## Why This Change

<!-- Explain why this approach was chosen. -->

## Training / Inference Impact

- [ ] Training behavior changes
- [ ] Inference behavior changes
- [ ] Config defaults changed
- [ ] Checkpoint compatibility may be affected
- [ ] Dataset or annotation assumptions changed
- [ ] Evaluation or metric behavior changed

## Validation

- [ ] Not run
- [ ] Smoke-tested locally
- [ ] Targeted unit or script validation run
- [ ] Full train/eval command run

### Commands / Evidence

<!-- Paste the exact commands, logs, or metrics used for validation. -->

## Risks And Compatibility Notes

<!-- Call out checkpoint loading, config inheritance, DDP/AMP, evaluator, or memory risks. -->

## Reviewer Guide

<!-- Mention the files or behaviors that deserve extra attention. -->

Recommended Codex review comment:

```text
@codex review
Please focus on:
- training/inference correctness
- AMP/DDP safety
- checkpoint loading compatibility
- config regressions
- evaluation metric changes
```

For local review packaging:

```bash
bash scripts/gen_review.sh
```
