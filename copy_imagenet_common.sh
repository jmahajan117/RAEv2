#!/bin/bash
# Copy the ImageNet folder from my /work/nvme dir to the shared bikv common dir so the whole group can use it.
# The copy is latency-bound (1.33M small files), so it runs one rsync per class dir, many at once.
# Usage (on a compute node, not the login node): ./copy_imagenet_common.sh [parallel_jobs]   (default 128)
# Resumable: re-running only copies files that are missing or changed.
set -euo pipefail

SRC=/work/nvme/bikv/jmahajan/imagenet-1k
DST=/work/nvme/bikv/common/imagenet-1k
JOBS=${1:-128}

mkdir -p "$DST/train" "$DST/validation"  # rsync only creates the last path component
cd "$SRC"
total=$(find train validation -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Copying $total class dirs from $SRC to $DST with $JOBS parallel rsyncs"
start=$(date +%s)

# -rlt (not -a): skip owner/group/perms so new files take the common dir's group and default ACL
# (group rwx), which keeps them readable by everyone in delta_bikv
find train validation -mindepth 1 -maxdepth 1 -type d -print0 \
  | xargs -0 -P "$JOBS" -I{} sh -c 'rsync -rlt "$1/" "$2/$1/" && echo "$1"' _ {} "$DST" \
  | awk -v t="$total" -v s="$start" '{ n++; if (n % 50 == 0 || n == t) { e = systime() - s; printf "%d / %d dirs (%ds elapsed)\n", n, t, e; fflush() } }'

# rsync killed mid-file (e.g. job time limit) leaves hidden temp files (.<name>.XXXXXX); the real
# file is recopied on the next run, so drop the leftovers before counting
lfs find "$DST" -type f -name '.*.??????' -print0 | xargs -0 -r rm -f

echo "Verifying file counts..."
for split in train validation; do
  a=$(lfs find "$SRC/$split" -type f | wc -l)
  b=$(lfs find "$DST/$split" -type f | wc -l)
  echo "  $split: source=$a copy=$b $([ "$a" = "$b" ] && echo OK || echo 'MISMATCH - re-run to finish')"
done
echo "Done in $(( $(date +%s) - start ))s"
