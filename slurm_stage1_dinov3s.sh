#!/bin/bash
#SBATCH --job-name="raev2_stage1_dinov3s"
#SBATCH --partition=ghx4            # DeltaAI GH200 partition (4x H100-class GPUs per node, 48h max)
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun spawns the per-GPU processes
#SBATCH --cpus-per-task=288         # whole node: 4x 72-core Grace CPUs
#SBATCH --gpus-per-node=4
#SBATCH --account=bikv-dtai-gh      # <- match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 48:00:00                 # ghx4 partition limit is 48h
#SBATCH -e ./logs/slurm-%A_%a.err
#SBATCH -o ./logs/slurm-%A_%a.out
#SBATCH --array=0-2   # one array task per variant below; each task gets its own node/4-GPU allocation

# Submit from the repo root (log paths are relative to it):  sbatch slurm_stage1_dinov3s.sh
# Resubmitting after a timeout resumes each run from its latest checkpoint.

export OMP_NUM_THREADS=1 # for delta
# DeltaAI is ARM (Grace); the site default module sets CXX to the Cray "CC" wrapper,
# which breaks torch.compile's Inductor C++ builds. Point it at real GCC.
export CXX=g++
export CC=gcc
NUM_GPUS=4

RAE_ROOT=/work/hdd/bibd/jmahajan/raev2
REF_NPZ=${RAE_ROOT}/data/imagenet-256-val.npz   # val reference for online rFID/PSNR/SSIM
export DINOV3_CKPT_DIR=${RAE_ROOT}/pretrained_models/encoders/dinov3
export WANDB_ENTITY=RenGen
export WANDB_PROJECT=RAEv2-Training

# Index must line up: CONFIGS[i] <-> RUN_NAMES[i]
CONFIGS=(
    "configs/stage1/training/dinov3s-k7-imagenet.yaml"
    "configs/stage1/training/dinov3s-k1-imagenet.yaml"
    "configs/stage1/training/dinov3s-k11-imagenet.yaml"
)
RUN_NAMES=(
    "dinov3s-k7-vitb-imagenet"
    "dinov3s-k1-vitb-imagenet"
    "dinov3s-k11-vitb-imagenet"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
export EXPERIMENT_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"  # checkpoint folder + W&B run name

echo "Array task ${SLURM_ARRAY_TASK_ID}: training on ${NUM_GPUS} GPU(s) with config ${CONFIG} (run: ${EXPERIMENT_NAME})"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rae

# The cudatoolkit module puts HPC SDK cuBLAS 12.9 on LD_LIBRARY_PATH; mixing its libcublasLt with
# torch's bundled cuBLAS 12.8 makes bf16 GEMMs fail with CUBLAS_STATUS_INVALID_VALUE.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/.*/math_libs' | paste -sd:)

if [ ! -f "${REF_NPZ}" ]; then
    echo "Missing ${REF_NPZ}; build it with scripts/data/make_imagenet_ref_npz.py first" >&2
    exit 1
fi

nvidia-smi

# srun doesn't inherit --cpus-per-task from sbatch; the env var passes it on (an explicit
# srun --cpus-per-task fails with "Error configuring interconnect" on DeltaAI)
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
srun bash -c 'echo "srun step: GPUs=${CUDA_VISIBLE_DEVICES} $(grep Cpus_allowed_list /proc/self/status)"'

srun torchrun --standalone --nproc_per_node="${NUM_GPUS}" src/train_stage1.py \
    --config "${CONFIG}" \
    --results-dir "${RAE_ROOT}/ckpts/stage1" \
    --precision bf16 \
    --compile \
    --wandb
TRAIN_STATUS=$?

# Final offline eval (PSNR / SSIM / rFID on the 50k val set) of the final checkpoint's EMA weights.
# Only after a successful run: with keep_last_checkpoints=1 the one remaining ep-*.pt is the final one.
if [ ${TRAIN_STATUS} -eq 0 ]; then
    FINAL_CKPT=$(ls "${RAE_ROOT}/ckpts/stage1/${EXPERIMENT_NAME}/checkpoints/"ep-*.pt | sort | tail -n 1)
    echo "Training done; final eval of ${FINAL_CKPT}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-final" srun torchrun --standalone --nproc_per_node="${NUM_GPUS}" src/offline_eval_stage1.py \
        --config "${CONFIG}" \
        --ckpt "${FINAL_CKPT}" \
        --precision bf16
else
    echo "Training exited with status ${TRAIN_STATUS}; skipping final eval" >&2
fi

# Report training failures to Slurm (otherwise the final echo makes the job look COMPLETED)
exit ${TRAIN_STATUS}
