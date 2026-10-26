#!/bin/bash
#SBATCH --job-name="raev2_eval_stage2_ig"
#SBATCH --partition=ghx4            # DeltaAI GH200 partition (4x H100-class GPUs per node)
#SBATCH --mem=220G                  # half a node (488G); 32 dataloader workers/GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun spawns the per-GPU processes
#SBATCH --cpus-per-task=144         # half a node: 2x 72-core Grace CPUs
#SBATCH --gpus-per-node=2
#SBATCH --account=bikv-dtai-gh      # <- match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 02:00:00                 # one 50k-sample, 100-step eval on 2 GPUs is ~1h
#SBATCH --exclude=gh014             # srun fails there with "Error configuring interconnect" (2026-10-19)
#SBATCH -e ./logs/slurm-%A_%a.err
#SBATCH -o ./logs/slurm-%A_%a.out
#SBATCH --array=0-5   # one array task per IG scale below; each task gets its own 2-GPU allocation

# Internal-guidance scale sweep: offline FID/IS (50k samples) of the final DiT^DH-S + IG checkpoint, one IG scale per
# array task, via eval_stage2_dinov3s.sh (sampling config configs/stage2/sampling/imagenet-dinov3s-k11-ditdhs-ig.yaml).
# Submit from the repo root (log paths are relative to it):  sbatch slurm_eval_stage2_ig.sh
# Results: /work/nvme/bikv/jmahajan/raev2/results/stage2/eval/imagenet-dinov3s-k11-vitb-ditdhs-ig-ig<scale>-steps<N>_ema.csv

# 1.0 = IG off (baseline at the same step count); 1.78 = the L sampling config's value
IG_SCALES=(1.0 1.25 1.5 1.78 2.0 2.5)
NUM_STEPS=100   # sampler steps for every scale (sampling config default; training-time evals used 50)
VARIANT=k11
MODEL=ditdhs-ig

export OMP_NUM_THREADS=1 # for delta
export IG_SCALE="${IG_SCALES[$SLURM_ARRAY_TASK_ID]}"
export NUM_STEPS
export NUM_GPUS=2

echo "Array task ${SLURM_ARRAY_TASK_ID}: ${MODEL} ${VARIANT}, ig.scale=${IG_SCALE}, num_steps=${NUM_STEPS} on ${NUM_GPUS} GPU(s)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rae

nvidia-smi

# srun doesn't inherit --cpus-per-task from sbatch; the env var passes it on (an explicit
# srun --cpus-per-task fails with "Error configuring interconnect" on DeltaAI)
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# eval_stage2_dinov3s.sh sets the cuBLAS/encoder/FID-cache env, applies IG_SCALE/NUM_STEPS to a derived config and
# names the run imagenet-dinov3s-<variant>-vitb-<model>-ig<scale>-steps<N>
srun ./eval_stage2_dinov3s.sh "${VARIANT}" "${MODEL}"
