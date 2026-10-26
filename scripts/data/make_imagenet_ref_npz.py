"""Build the reference NPZ for online stage-1 reconstruction eval from an ImageNet folder.

Uses the same dataset ordering and ADM center crop as the `type: folder` eval
loader, so reconstructions align index-by-index with the reference images
(required for PSNR/SSIM; rFID only needs the distribution).

Usage:
    uv run python scripts/data/make_imagenet_ref_npz.py \
        --data-dir /work/nvme/bikv/jmahajan/imagenet-1k \
        --out /work/hdd/bibd/jmahajan/raev2/data/imagenet-256-val.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.imagenet_folder_dataset import ImageNetFolderDataset  # noqa: E402
from utils.train_utils import center_crop_arr  # noqa: E402


def to_uint8(img, image_size):
    return np.asarray(center_crop_arr(img, image_size), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = ImageNetFolderDataset(
        args.data_dir, split=args.split, transform=partial(to_uint8, image_size=args.image_size)
    )
    loader = DataLoader(dataset, batch_size=250, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda b: np.stack([x[0] for x in b]))

    arr = np.empty((len(dataset), args.image_size, args.image_size, 3), dtype=np.uint8)
    i = 0
    for batch in tqdm(loader, desc="Building reference NPZ"):
        arr[i:i + len(batch)] = batch
        i += len(batch)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"  # write-then-rename so concurrent runs never see a partial file
    np.savez(tmp, arr_0=arr)
    os.replace(tmp, args.out)
    print(f"Saved {arr.shape} to {args.out}")


if __name__ == "__main__":
    main()
