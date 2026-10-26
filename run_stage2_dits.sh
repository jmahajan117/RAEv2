#!/bin/bash
# Stage-2 training: DiT-S family on DINOv3-S RAE latents (ViT-B decoder), ImageNet-1k 256px (class-conditional).
# Usage: ./run_stage2_dits.sh [k11] [ditdhs-ig|ditdhs|dits]   (defaults k11, ditdhs-ig;
#        uses configs/stage2/training/imagenet-dinov3s-<variant>-<model>.yaml)
# Run on a GPU node (e.g. inside run_ai_interactive.sh). Uses all visible GPUs unless NUM_GPUS is set.
# Checkpoints land in $OUT_ROOT/ckpts/stage2/$EXPERIMENT_NAME. By default each launch deletes that run and starts
# over; FRESH=0 ./run_stage2_dits.sh resumes from the latest checkpoint instead.
# Expects the `rae` conda env to already be active, stats.pt (./stats_stage1_dinov3s.sh <variant>), and the
# ViT-B decoder.pt extracted from the stage-1 run (see the error message below).
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)
# DeltaAI's default CXX is the Cray "CC" wrapper, which breaks torch.compile's Inductor C++ builds.
export CXX=g++
export CC=gcc

############ Experiment settings ############
VARIANT=${1:-k11}
# Stage-2 model: ditdhs-ig (DiT^DH-S + internal guidance), ditdhs (DiT^DH-S), dits (plain DiT-S)
MODEL=${2:-ditdhs-ig}
# Run name: checkpoint folder + W&B run. Leave empty for imagenet-dinov3s-<variant>-vitb-<model>.
EXP_NAME="test-$MODEL-dinov3s-$VARIANT-vitb"
#############################################

RAE_ROOT=/work/hdd/bibd/jmahajan/raev2      # inputs: stage-1 decoder/stats, encoder weights
OUT_ROOT=/work/nvme/bikv/jmahajan/raev2     # stage-2 outputs: checkpoints, eval shards/CSVs, FID stats cache
CONFIG=configs/stage2/training/imagenet-dinov3s-$VARIANT-$MODEL.yaml
STATS=$RAE_ROOT/pretrained_models/stage1/imagenet/dinov3s-$VARIANT/stats.pt
DECODER=$RAE_ROOT/pretrained_models/stage1/imagenet/dinov3s-$VARIANT-vitb/decoder.pt  # must match the config

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_NAME:-imagenet-dinov3s-$VARIANT-vitb-$MODEL}}
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
export NANOGEN_EVALS_CACHE_DIR=$OUT_ROOT/cache/nanogen-evals  # FID reference stats download (default is ~/.cache)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

if [ ! -f "$CONFIG" ]; then
  echo "Missing $CONFIG" >&2
  exit 1
fi
if [ ! -f "$STATS" ]; then
  echo "Missing $STATS; run ./stats_stage1_dinov3s.sh $VARIANT first" >&2
  exit 1
fi
if [ ! -f "$DECODER" ]; then
  echo "Missing $DECODER; extract it from the stage-1 run:" >&2
  echo "  python scripts/stage1/extract_decoder.py --config configs/stage1/training/dinov3s-$VARIANT-imagenet.yaml \\" >&2
  echo "    --ckpt $RAE_ROOT/ckpts/stage1/dinov3s-$VARIANT-vitb-imagenet/checkpoints/<final ep-*.pt> --use-ema --out $DECODER" >&2
  exit 1
fi

# FRESH=1 (default) deletes this run's checkpoints and eval CSV first, so training starts over; FRESH=0 resumes.
# Stop any torchrun still training this run before using it. (The W&B run is not touched.)
if [ "${FRESH:-1}" = 1 ]; then
  echo "FRESH=1: deleting $OUT_ROOT/ckpts/stage2/$EXPERIMENT_NAME and its eval CSV"
  rm -rf "$OUT_ROOT/ckpts/stage2/$EXPERIMENT_NAME"
  rm -f "$OUT_ROOT/results/stage2/eval/${EXPERIMENT_NAME}_ema.csv"
fi

# W&B logging is on by default; set NO_WANDB=1 to disable
export WANDB_ENTITY=${WANDB_ENTITY:-RenGen}
export WANDB_PROJECT=${WANDB_PROJECT:-RAEv2-Training}
WANDB_ARGS=(--wandb)
if [ "${NO_WANDB:-0}" = 1 ]; then
  WANDB_ARGS=()
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  src/train.py \
  --config "$CONFIG" \
  --results-dir "$OUT_ROOT/ckpts/stage2" \
  --precision bf16 \
  --compile \
  "${WANDB_ARGS[@]}"
