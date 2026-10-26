#!/bin/bash
#SBATCH --job-name="raev2_stage2_dits"
#SBATCH --partition=ghx4            # DeltaAI GH200 partition (4x H100-class GPUs per node, 48h max)
#SBATCH --mem=400G                   # 32 dataloader workers/GPU use ~165G of process memory (node has 488G)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun spawns the per-GPU processes
#SBATCH --cpus-per-task=288         # whole node: 4x 72-core Grace CPUs
#SBATCH --gpus-per-node=4
#SBATCH --account=bikv-dtai-gh      # <- match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 48:00:00                 # ghx4 partition limit is 48h
#SBATCH -e ./logs/slurm-%A_%a.err
#SBATCH -o ./logs/slurm-%A_%a.out
#SBATCH --array=0   # one array task per config below; each task gets its own node/4-GPU allocation

# Submit from the repo root (log paths are relative to it):  sbatch slurm_stage2_dits.sh
# Resubmitting after a timeout resumes each run from its latest checkpoint.
# Needs stats.pt first: ./stats_stage1_dinov3s.sh k11 (on a GPU node), plus the ViT-B decoder.pt
# extracted from the stage-1 run (scripts/stage1/extract_decoder.py --use-ema).

export OMP_NUM_THREADS=1 # for delta
# DeltaAI is ARM (Grace); the site default module sets CXX to the Cray "CC" wrapper,
# which breaks torch.compile's Inductor C++ builds. Point it at real GCC.
export CXX=g++
export CC=gcc
NUM_GPUS=4

RAE_ROOT=/work/hdd/bibd/jmahajan/raev2      # inputs: stage-1 decoder/stats, encoder weights
OUT_ROOT=/work/nvme/bikv/jmahajan/raev2   # stage-2 outputs: checkpoints, eval shards/CSVs, FID stats cache
export DINOV3_CKPT_DIR=${RAE_ROOT}/pretrained_models/encoders/dinov3
export NANOGEN_EVALS_CACHE_DIR=${OUT_ROOT}/cache/nanogen-evals  # FID reference stats download (default is ~/.cache)
export WANDB_ENTITY=RenGen
export WANDB_PROJECT=RAEv2-Training

# Index must line up: CONFIGS[i] <-> RUN_NAMES[i]
# Previous run: configs/stage2/training/imagenet-dinov3s-k11-dits.yaml <-> imagenet-dinov3s-k11-vitb-dits (plain DiT-S)
CONFIGS=(
    "configs/stage2/training/imagenet-dinov3s-k11-ditdhs-ig.yaml"
)
RUN_NAMES=(
    "imagenet-dinov3s-k11-vitb-ditdhs-ig"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
export EXPERIMENT_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"  # checkpoint folder + W&B run name

echo "Array task ${SLURM_ARRAY_TASK_ID}: training on ${NUM_GPUS} GPU(s) with config ${CONFIG} (run: ${EXPERIMENT_NAME})"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rae

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd:)

STATS=${RAE_ROOT}/pretrained_models/stage1/imagenet/dinov3s-k11/stats.pt
if [ ! -f "${STATS}" ]; then
    echo "Missing ${STATS}; run ./stats_stage1_dinov3s.sh k11 on a GPU node first" >&2
    exit 1
fi
DECODER=${RAE_ROOT}/pretrained_models/stage1/imagenet/dinov3s-k11-vitb/decoder.pt  # must match the config
if [ ! -f "${DECODER}" ]; then
    echo "Missing ${DECODER}; extract it from ckpts/stage1/dinov3s-k11-vitb-imagenet with scripts/stage1/extract_decoder.py --use-ema" >&2
    exit 1
fi

nvidia-smi

# srun doesn't inherit --cpus-per-task from sbatch; the env var passes it on (an explicit
# srun --cpus-per-task fails with "Error configuring interconnect" on DeltaAI)
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
srun bash -c 'echo "srun step: GPUs=${CUDA_VISIBLE_DEVICES} $(grep Cpus_allowed_list /proc/self/status)"'

srun torchrun --standalone --nproc_per_node="${NUM_GPUS}" src/train.py \
    --config "${CONFIG}" \
    --results-dir "${OUT_ROOT}/ckpts/stage2" \
    --precision bf16 \
    --compile \
    --wandb
TRAIN_STATUS=$?

# Final offline eval (FID/IS on the full val-size sample set) of the final checkpoint's EMA weights.
# Only after a successful run: with keep_last_checkpoints=1 the one remaining ep-*.pt is the final one.
if [ ${TRAIN_STATUS} -eq 0 ]; then
    FINAL_CKPT=$(ls "${OUT_ROOT}/ckpts/stage2/${EXPERIMENT_NAME}/checkpoints/"ep-*.pt | sort | tail -n 1)
    echo "Training done; final eval of ${FINAL_CKPT}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-final" srun torchrun --standalone --nproc_per_node="${NUM_GPUS}" src/offline_eval.py \
        --config "${CONFIG}" \
        --ckpt "${FINAL_CKPT}" \
        --precision bf16
else
    echo "Training exited with status ${TRAIN_STATUS}; skipping final eval" >&2
fi

# Report training failures to Slurm (otherwise the final echo makes the job look COMPLETED)
exit ${TRAIN_STATUS}
