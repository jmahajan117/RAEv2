#!/bin/bash
#SBATCH --job-name="raev2_hold_node"
#SBATCH --partition=ghx4            # DeltaAI GH200 partition (4x H100-class GPUs per node, 48h max)
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288         # whole node: 4x 72-core Grace CPUs
#SBATCH --gpus-per-node=4
#SBATCH --account=bibd-dtai-gh      # bibd-dtai-gh currently has better priority: sbatch --account=bibd-dtai-gh ...
#SBATCH --no-requeue
#SBATCH -t 48:00:00                 # ghx4 partition limit is 48h
#SBATCH --mail-type=BEGIN,END       # email when the node is ready
#SBATCH -e ./logs/hold-%j.err
#SBATCH -o ./logs/hold-%j.out

# Holds a full ghx4 node for 48h so you can ssh in and work interactively.
#   Submit:    sbatch slurm_hold_node_48h.sh
#   Wait:      squeue -u $USER --start          (estimated start time)
#   Connect:   ssh $(cat logs/hold_node.txt)     (once it is RUNNING)
#         or:  srun --jobid=<JOBID> --overlap --pty bash
#   Release:   scancel <JOBID>                   (free the node early when done)

echo "${SLURMD_NODENAME}" > logs/hold_node.txt
echo "[$(date)] Job ${SLURM_JOB_ID} running on ${SLURMD_NODENAME}"
echo "Connect with:  ssh ${SLURMD_NODENAME}"
echo "         or:   srun --jobid=${SLURM_JOB_ID} --overlap --pty bash"
nvidia-smi -L

sleep infinity
