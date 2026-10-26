#!/bin/bash
# Offline stage-2 eval (FID / IS on 50k generated samples vs ImageNet val) of a trained DiT-S-family model on
# DINOv3-S RAE latents, using a sampling config (guidance/sampler settings) from configs/stage2/sampling.
# Usage: ./eval_stage2_dinov3s.sh [k11] [ditdhs-ig]   (defaults k11, ditdhs-ig;
#        uses configs/stage2/sampling/imagenet-dinov3s-<variant>-<model>.yaml)
# Overrides for sweeps (written to a derived config, the sampling config is not edited):
#   IG_SCALE=2.0 ./eval_stage2_dinov3s.sh      # guidance.ig.scale (1.0 = IG off)
#   NUM_STEPS=50 ./eval_stage2_dinov3s.sh      # sampler.num_steps
# Run on a GPU node (e.g. inside run_ai_interactive.sh) with the `rae` conda env active. Uses all visible GPUs
# unless NUM_GPUS is set.
# Results: $OUT_ROOT/results/stage2/eval/<EXPERIMENT_NAME>_ema.csv
set -euo pipefail
cd "$(dirname "$0")"

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd: || true)

VARIANT=${1:-k11}
MODEL=${2:-ditdhs-ig}
RAE_ROOT=/work/hdd/bibd/jmahajan/raev2      # inputs: stage-1 decoder/stats, encoder weights
OUT_ROOT=/work/nvme/bikv/jmahajan/raev2     # stage-2 checkpoints, eval shards/CSVs, FID stats cache
CONFIG=configs/stage2/sampling/imagenet-dinov3s-$VARIANT-$MODEL.yaml
CKPT=$OUT_ROOT/ckpts/stage2/imagenet-dinov3s-$VARIANT-vitb-$MODEL/checkpoints/ep-0000080.pt  # final checkpoint (EMA weights)

export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
export NANOGEN_EVALS_CACHE_DIR=$OUT_ROOT/cache/nanogen-evals  # FID reference stats download (default is ~/.cache)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

for f in "$CONFIG" "$CKPT"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f" >&2
    exit 1
  fi
done

# Apply IG_SCALE / NUM_STEPS to a copy of the config, and read back the effective values for the run name
DERIVED=$OUT_ROOT/tmp/eval_stage2_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_$$.yaml  # unique across array tasks/nodes
mkdir -p "$(dirname "$DERIVED")"
trap 'rm -f "$DERIVED"' EXIT
read -r EFF_IG EFF_STEPS < <(python - "$CONFIG" "$DERIVED" "${IG_SCALE:-}" "${NUM_STEPS:-}" <<'EOF'
import sys
from omegaconf import OmegaConf
src, dst, ig_scale, num_steps = sys.argv[1:]
cfg = OmegaConf.load(src)
if ig_scale:
    cfg.guidance.ig.scale = float(ig_scale)
if num_steps:
    cfg.sampler.num_steps = int(num_steps)
OmegaConf.save(cfg, dst)
ig = cfg.guidance.get("ig")
print(ig.scale if ig is not None else 1.0, cfg.sampler.num_steps)
EOF
)

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-imagenet-dinov3s-$VARIANT-vitb-$MODEL-ig$EFF_IG-steps$EFF_STEPS}
echo "Evaluating $CKPT with ig.scale=$EFF_IG, num_steps=$EFF_STEPS on $NUM_GPUS GPU(s) (run: $EXPERIMENT_NAME)"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  src/offline_eval.py \
  --config "$DERIVED" \
  --ckpt "$CKPT" \
  --precision bf16
