#!/bin/bash
# Latent normalization stats (per-dim mean/var) of a DINOv3-S RAE encoder over ImageNet train,
# needed by stage 2 (stage_1.normalization_stat_path). Encoder-only, so no stage-1 checkpoint needed.
# Usage: ./stats_stage1_dinov3s.sh [k1|k7|k11]   (default k11)
# Run on a GPU node (e.g. inside run_ai_interactive.sh) with the `rae` conda env active; uses all visible GPUs.
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)

VARIANT=${1:-k11}
RAE_ROOT=/work/hdd/bibd/jmahajan/raev2
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3

torchrun --standalone --nproc_per_node="$(nvidia-smi -L | wc -l)" \
  scripts/stage1/compute_encoder_stats.py \
  --config configs/stage1/training/dinov3s-$VARIANT-imagenet.yaml \
  --data-path /work/nvme/bikv/jmahajan/imagenet-1k/train \
  --batch-size 256 \
  --num-workers 16 \
  --output-path $RAE_ROOT/pretrained_models/stage1/imagenet/dinov3s-$VARIANT/stats.pt
