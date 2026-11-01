#!/usr/bin/env bash
### Check hostname

if [ "$(hostname)" != "vision-c21.cs.illinois.edu" ]; then
    DATA_DIR="/scratch-ssd/common/imagenet-1k"
elif [ "$(hostname)" != "vision-c22.cs.illinois.edu" ]; then
    DATA_DIR="/scratch-spin/common/imagenet-1k"
elif [ "$(hostname)" != "vision-c23.cs.illinois.edu" ]; then
    DATA_DIR="/scratch-spin/common/imagenet-1k"
else
    echo "Unknown hostname"
    exit 1
fi


hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir $DATA_DIR
