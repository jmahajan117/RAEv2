#!/bin/bash
# Interactive 4-GPU session on DeltaAI ghx4 nodes (max 2 hr on ghx4-interactive).
# Lands a shell on a compute node; use NUM_GPUS=4 for torchrun inside.
srun --account=bdha-dtai-gh --time=2:00:00 --partition=ghx4-interactive \
  --nodes=1 --gpus-per-node=2 --ntasks=1 \
  --cpus-per-task=144 \
  --pty bash