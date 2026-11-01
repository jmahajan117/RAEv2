set -eo pipefail   # no -u: conda's activate scripts read unset variables
cd "$(dirname "$0")"   # repo root, so environment.yml resolves wherever this is run from

## Download and install Miniconda at the home directory (-b: no prompts, -p: install to ~/miniconda3)
curl -o "$HOME/Miniconda3-latest-Linux-x86_64.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash "$HOME/Miniconda3-latest-Linux-x86_64.sh" -b -p "$HOME/miniconda3"

## Add conda to the PATH environment variable and make conda available in the current shell session
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda init bash   # so new shells have conda too

## Create conda environment (environment.yml uses the defaults channels, whose ToS must be accepted first)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda env create -f environment.yml
conda activate rengen


mkdir -p ~/.cache/torch/hub/checkpoints

# DINOv3 ViT-S/16 backbone: FREN refuses to start without it on disk (REN/encoders/fren.py:377).
# Copied from your Delta home; set DELTA_USER to skip the prompt.
# DELTA_USER=${DELTA_USER:-}
# while [ -z "$DELTA_USER" ]; do
#   read -rp "Delta username: " DELTA_USER
# done
# rsync -avP "$DELTA_USER@dt-login.delta.ncsa.illinois.edu":'~/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth' \
#   ~/.cache/torch/hub/checkpoints/


### Rsync the imagenet statistics file from Delta to the local machine, so that the FREN model can be initialized properly.
# Per-variant latent normalization stats (stats.pt from ./stats_stage1_dinov3s.sh), read at RAE init via
# normalization_stat_path. Lands at the repo-relative pretrained_models/stage1/imagenet/<variant>/stats.pt.
# /work is shared across Delta and DeltaAI; set DELTA_USER / DELTA_HOST to skip the prompt / change login node.
DELTA_USER=${DELTA_USER:-}
while [ -z "$DELTA_USER" ]; do
  read -rp "Delta username: " DELTA_USER
done
DELTA_HOST=${DELTA_HOST:-dtai-login.delta.ncsa.illinois.edu}
mkdir -p pretrained_models/stage1/imagenet
rsync -avP --prune-empty-dirs --include='*/' --include='stats.pt' --exclude='*' \
  "$DELTA_USER@$DELTA_HOST":/work/hdd/bibd/jmahajan/raev2/pretrained_models/stage1/imagenet/ \
  pretrained_models/stage1/imagenet/

### Rsync the 512px ImageNet val reference images for rFID (50000x512x512x3 uint8, ~39 GB; --partial resumes).
# Same order + center crop as the folder eval loader; point the config's eval reference_npz here.
mkdir -p data/imagenet-512
rsync -avP --partial "$DELTA_USER@$DELTA_HOST":/work/hdd/bibd/common-ckpts/imagenet-512-val.npz \
  data/imagenet-512/

