#!/bin/bash
# Stage-1 RAE training: DINOv3-S16 on ImageNet folder at /work/nvme/bikv/jmahajan/imagenet-1k.
# Usage: ./run_stage1_dinov3s.sh [k1|k7|k11]   (default k7)
# Run on a GPU node (e.g. inside run_ai_interactive.sh). Uses all visible GPUs unless NUM_GPUS is set.
# Checkpoints land in $RAE_ROOT/ckpts/stage1/$EXPERIMENT_NAME; re-running resumes from the latest one.
# Expects the `rae` conda env to already be active.
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)

############ Experiment settings ############
VARIANT=${1:-k7}
# Run name: checkpoint folder + W&B run. Leave empty for dinov3s-<variant>-vitb-imagenet (ViT-B decoder).
EXP_NAME="test-dinov3s"
#############################################

RAE_ROOT=/work/hdd/bibd/jmahajan/raev2
IMAGENET_DIR=/work/nvme/bikv/jmahajan/imagenet-1k
CONFIG=configs/stage1/training/dinov3s-$VARIANT-imagenet.yaml
REF_NPZ=$RAE_ROOT/data/imagenet-256-val.npz

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_NAME:-dinov3s-$VARIANT-vitb-imagenet}}
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

# Reference val images for online rFID/PSNR/SSIM (one-time, ~10GB)
if [ ! -f "$REF_NPZ" ]; then
  python scripts/data/make_imagenet_ref_npz.py --data-dir "$IMAGENET_DIR" --out "$REF_NPZ"
fi

# W&B logging is on by default; set NO_WANDB=1 to disable
export WANDB_ENTITY=${WANDB_ENTITY:-RenGen}
export WANDB_PROJECT=${WANDB_PROJECT:-RAEv2-Training}
WANDB_ARGS=(--wandb)
if [ "${NO_WANDB:-0}" = 1 ]; then
  WANDB_ARGS=()
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  src/train_stage1.py \
  --config "$CONFIG" \
  --results-dir "$RAE_ROOT/ckpts/stage1" \
  --precision bf16 \
  --compile \
  "${WANDB_ARGS[@]}"
