#!/bin/bash
# Stage-1 RAE training: DINOv3-S16 on ImageNet folder at /work/nvme/bikv/jmahajan/imagenet-1k.
# Usage: ./run_stage1_dinov3s.sh [k1|k7|k11|k11-512]   (default k11-512: native 512, 32x32 tokens)
# Run on a GPU node (e.g. inside run_ai_interactive.sh). Uses all visible GPUs unless NUM_GPUS is set.
# Checkpoints land in $RAE_ROOT/ckpts/stage1/$EXPERIMENT_NAME; re-running resumes from the latest one.
# FRESH=1 (default) deletes this run's checkpoints, logs, eval shards/CSVs and local W&B files first, so a
#   test run starts from scratch as a new W&B run. Only run names starting with "test" are wiped; others
#   resume as usual. FRESH=0 ./run_stage1_dinov3s.sh resumes a test run instead.
# Expects the `rae` conda env to already be active.
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)

############ Experiment settings ############
VARIANT=${1:-k11-512}
# Run name: checkpoint folder + W&B run. Leave empty for dinov3s-<variant>-vitb-imagenet (ViT-B decoder).
EXP_NAME="test-dinov3s"
#############################################

RAE_ROOT=/work/hdd/bibd/jmahajan/raev2
IMAGENET_DIR=/work/nvme/bikv/jmahajan/imagenet-1k
CONFIG=configs/stage1/training/dinov3s-$VARIANT-imagenet.yaml
RES=$(grep -oP '^  image_size: \K\d+' "$CONFIG")
# Must match the config's eval reference_npz (512 lives in the shared common-ckpts dir)
REF_NPZ=$RAE_ROOT/data/imagenet-$RES-val.npz
[ "$RES" = 512 ] && REF_NPZ=/work/hdd/bibd/common-ckpts/imagenet-512-val.npz

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_NAME:-dinov3s-$VARIANT-vitb-imagenet}}
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

if [ "${FRESH:-1}" = 1 ] && [[ "$EXPERIMENT_NAME" != test* ]]; then
  echo "FRESH: $EXPERIMENT_NAME is not a test run; keeping its files (resuming)"
elif [ "${FRESH:-1}" = 1 ]; then
  EXP_DIR=$RAE_ROOT/ckpts/stage1/$EXPERIMENT_NAME
  EVAL_TMP_ROOT=$(grep -oP '^  shared_tmpdir: \K\S+' "$CONFIG" || true)
  CLEAN=("$EXP_DIR" experiments/"${RAE_USER:-jas}"/evals/stage1/"$EXPERIMENT_NAME"_*.csv)
  [ -n "$EVAL_TMP_ROOT" ] && CLEAN+=("$EVAL_TMP_ROOT/stage1/$EXPERIMENT_NAME")
  if [ -f "$EXP_DIR/wandb_run_id.txt" ]; then
    CLEAN+=(wandb/run-*-"$(cat "$EXP_DIR/wandb_run_id.txt")")
  fi
  for p in "${CLEAN[@]}"; do
    if [ -e "$p" ]; then
      echo "FRESH: removing $p"
      rm -rf -- "$p"
    fi
  done
fi

# Reference val images for online rFID/PSNR/SSIM (one-time, ~10GB at 256 / ~40GB at 512)
if [ ! -f "$REF_NPZ" ]; then
  python scripts/data/make_imagenet_ref_npz.py --data-dir "$IMAGENET_DIR" --image-size "$RES" --out "$REF_NPZ"
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
