#!/bin/bash
# Offline stage-1 eval (PSNR / SSIM / rFID on the 50k ImageNet val set) of a trained DINOv3-S RAE.
# Usage: ./eval_stage1_dinov3s.sh [k1|k7|k11] [256|512]   (default k11 256)
#   512: runs the 256-trained RAE natively at 512 (32x32 tokens, re-built decoder sin-cos pos-embed)
#        against a 512 center-crop reference (shared copy in /work/hdd/bibd/common-ckpts; else built
#        on first use, ~40GB in $RAE_ROOT/data).
# Run on a 1-GPU node (e.g. inside run_ai_interactive.sh) with the `rae` conda env active.
# Results: $RAE_ROOT/results/stage1/eval/<EXPERIMENT_NAME>_ema_imagenet.csv
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)

VARIANT=${1:-k11}
RES=${2:-512}
RAE_ROOT=/work/hdd/bibd/jmahajan/raev2
CONFIG=configs/stage1/training/dinov3s-$VARIANT-imagenet.yaml
CKPT=$RAE_ROOT/ckpts/stage1/dinov3s-$VARIANT-vitb-imagenet/checkpoints/ep-0000016.pt  # final checkpoint (EMA weights); the config now builds a ViT-B decoder

export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3

RES_ARGS=()
if [ "$RES" != 256 ]; then
  REF_NPZ=/work/hdd/bibd/common-ckpts/imagenet-$RES-val.npz  # shared copy (512 exists; same order/crop as the eval loader)
  [ -f "$REF_NPZ" ] || REF_NPZ=$RAE_ROOT/data/imagenet-$RES-val.npz
  if [ ! -f "$REF_NPZ" ]; then
    python scripts/data/make_imagenet_ref_npz.py \
      --data-dir /work/nvme/bikv/jmahajan/imagenet-1k --image-size "$RES" --out "$REF_NPZ"
  fi
  RES_ARGS=(--resolution "$RES" --reference-npz "$REF_NPZ")
  export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3s-$VARIANT-vitb-imagenet-final-res$RES}
fi
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3s-$VARIANT-vitb-imagenet-final}

torchrun --standalone --nproc_per_node=1 \
  src/offline_eval_stage1.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  "${RES_ARGS[@]}"
