#!/usr/bin/env bash
# Evaluate while excluding "negative" classes — analogue of WeDetect's
# test_exclude_negative.py.
#
# Usage:
#   bash scripts/eval_exclude_negative.sh \
#     <config_file> <init_ckpt> <output_dir> \
#     [--exclude-class-names "name_a,name_b,..."] \
#     [--exclude-from-file path/to/list.txt] \
#     [extra --opts...]
#
# If neither --exclude-class-names nor --exclude-from-file is passed,
# the WeDetect TCT_NGC negative list baked into the Python entry script
# is used (see DEFAULT_NEGATIVE_CLASS_NAMES in tools/eval_exclude_negative.py).
# set -x

root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino
time=$(date "+%Y%m%d-%H%M%S")

config_file=$1
init_ckpt=$(realpath $2)
output_dir=$3
shift 3 || true
dataset=$(basename $config_file | sed 's/.*_\(.*\)\.py/\1/')

# Same env plumbing as scripts/eval.sh — keeps DETECTRON2_DATASETS,
# HF_HOME, and transformers offline settings consistent.
export DETECTRON2_DATASETS="$root_dir/datas/"
export HF_HOME="$root_dir/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Distributed Testing on $dataset (excluding negative classes)"
evaluation_dir="$output_dir/eval_excluded_${dataset}_$time"
mkdir -p $evaluation_dir
cd $code_dir
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./tools/eval_exclude_negative.py \
    --config-file $config_file \
    --eval-only \
    --resume \
    "$@" \
    train.init_checkpoint=$init_ckpt \
    train.output_dir=$output_dir \
    dataloader.evaluator.output_dir="$evaluation_dir" \
    | tee $evaluation_dir/eval_$time.log
