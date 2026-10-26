#!/bin/bash
# Offline stage-1 eval (PSNR / SSIM / rFID on the 50k ImageNet val set) of a trained DINOv3-S RAE.
# Usage: ./eval_stage1_dinov3s.sh [k1|k7|k11]   (default k11)
# Run on a 1-GPU node (e.g. inside run_ai_interactive.sh) with the `rae` conda env active.
# Results: $RAE_ROOT/results/stage1/eval/<EXPERIMENT_NAME>_ema_imagenet.csv
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)

VARIANT=${1:-k11}
RAE_ROOT=/work/hdd/bibd/jmahajan/raev2
CONFIG=configs/stage1/training/dinov3s-$VARIANT-imagenet.yaml
CKPT=$RAE_ROOT/ckpts/stage1/dinov3s-$VARIANT-vitb-imagenet/checkpoints/ep-0000016.pt  # final checkpoint (EMA weights); the config now builds a ViT-B decoder

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3s-$VARIANT-vitb-imagenet-final}
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3

torchrun --standalone --nproc_per_node=1 \
  src/offline_eval_stage1.py \
  --config "$CONFIG" \
  --ckpt "$CKPT"
