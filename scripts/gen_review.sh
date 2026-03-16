#!/usr/bin/env bash
# gen_review.sh — Generate a review package to paste into ChatGPT/Codex
#
# Usage:
#   bash scripts/gen_review.sh                  # diff against main
#   bash scripts/gen_review.sh <base_branch>    # diff against specific branch
#   bash scripts/gen_review.sh --staged         # only staged changes
#   bash scripts/gen_review.sh --unstaged       # only unstaged changes (working tree)
#
# Output is printed to stdout. Redirect to a file or pipe to clipboard:
#   bash scripts/gen_review.sh > review.txt
#   bash scripts/gen_review.sh | xclip -selection clipboard   # Linux
#   bash scripts/gen_review.sh | pbcopy                       # macOS

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKLIST="$REPO_ROOT/.github/REVIEW_CHECKLIST.md"

# Determine diff mode
DIFF_CMD=()
DESCRIPTION=""

if [[ "${1:-}" == "--staged" ]]; then
    DIFF_CMD=(git diff --cached)
    DESCRIPTION="Staged changes (not yet committed)"
elif [[ "${1:-}" == "--unstaged" ]]; then
    DIFF_CMD=(git diff)
    DESCRIPTION="Unstaged working tree changes"
else
    BASE="${1:-main}"
    # Try the base branch; fall back to HEAD~1 if it doesn't exist
    if git rev-parse --verify "$BASE" >/dev/null 2>&1; then
        DIFF_CMD=(git diff "$BASE"...HEAD)
        DESCRIPTION="Changes from $BASE to HEAD"
    else
        echo "Warning: branch '$BASE' not found, falling back to HEAD~1" >&2
        DIFF_CMD=(git diff HEAD~1)
        DESCRIPTION="Changes in latest commit"
    fi
fi

cd "$REPO_ROOT"

DIFF_OUTPUT=$("${DIFF_CMD[@]}" 2>/dev/null || true)

if [[ -z "$DIFF_OUTPUT" ]]; then
    echo "No changes found ($DESCRIPTION)." >&2
    exit 0
fi

# Count stats
STAT_OUTPUT=$("${DIFF_CMD[@]}" --stat 2>/dev/null || true)
FILE_COUNT=$(echo "$STAT_OUTPUT" | grep -c '|' || true)

# Build the review package
cat <<'PROMPT_HEADER'
You are reviewing code changes for OV-DINO, an open-vocabulary object detection model.
Architecture: Swin Transformer (visual) + BERT (language) + DINO transformer (detection) with Language-Aware Selective Fusion.
Framework: Detectron2 + detrex, PyTorch DDP, optional AMP.

Please review the diff below. Structure your review using the checklist provided.
For each issue found, quote the relevant line(s) and explain the concern.
Rate overall risk: LOW / MEDIUM / HIGH.

PROMPT_HEADER

echo "--- CHANGE SUMMARY ---"
echo "$DESCRIPTION"
echo "Files changed: $FILE_COUNT"
echo ""
echo "$STAT_OUTPUT"
echo ""

echo "--- DIFF ---"
echo '```diff'
echo "$DIFF_OUTPUT"
echo '```'
echo ""

if [[ -f "$CHECKLIST" ]]; then
    echo "--- REVIEW CHECKLIST ---"
    cat "$CHECKLIST"
    echo ""
fi

echo "--- END OF REVIEW PACKAGE ---"
