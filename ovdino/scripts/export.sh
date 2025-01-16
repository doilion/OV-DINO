#!/usr/bin/env bash
# set -x

# project config
root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino
time=$(date "+%Y%m%d-%H%M%S")

config_file=$1
init_ckpt=$(realpath $2)
output_dir=$3
sample_image=${4:-""}
sample_categories=${5:-""}
dataset=$(basename $config_file | sed 's/.*_\(.*\)\.py/\1/')

export CUDA_HOME="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/wanghao/softs/cudas/cuda11.6"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo -e "$log_format cuda version:\n$(nvcc -V)"

# env config
export DETECTRON2_DATASETS="$root_dir/datas/"
export HF_HOME="$root_dir/inits/hgfc"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Exporting model to $output_dir"
export_dir="${output_dir}_${time}"
mkdir -p $export_dir
cd $code_dir
echo "Export dir: $export_dir"

# addtional export args
export_args=""
if [ ! -z "$sample_image" ]; then
    export_args="$export_args --sample-image $sample_image"
fi
if [ ! -z "$sample_categories" ]; then
    export_args="$export_args --sample-categories $sample_categories"
fi
echo "Export args: $export_args"

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./tools/deploy/export_model.py \
    --format "onnx" \
    --export-method "tracing" \
    --output $export_dir \
    $export_args \
    --config-file $config_file \
    --compare-export \
    train.init_checkpoint=$init_ckpt \
    train.output_dir=$export_dir \
    dataloader.evaluator.output_dir="$export_dir"