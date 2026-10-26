"""
ImageNet dataset loader for raw ImageFolder-style directories.

Expects `<data_dir>/<split_dir>/<class_dir>/<image>` where class dirs are either
WordNet ids (`n01440764`) or HF-style lemma strings (`tench, Tinca tinca`; a
duplicated lemma's second occurrence is suffixed with `2`, e.g. `crane2`).
Labels follow the standard ImageNet-1k (sorted wnid) ordering so they match
the Arrow-format ImageNetHFDataset.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .imagenet_classes import IMAGENET_CLASSES

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# split name -> candidate sub-directory names
SPLIT_DIRS = {
    "train": ("train",),
    "val": ("val", "validation"),
}


def _class_dir_to_label() -> dict:
    """Map wnid and lemma-string class dir names to standard ImageNet-1k indices."""
    from timm.data import ImageNetInfo

    info = ImageNetInfo("imagenet-1k")
    mapping = {}
    for idx in range(info.num_classes()):
        mapping[info.index_to_label_name(idx)] = idx
        lemma = info.index_to_description(idx)
        mapping[lemma if lemma not in mapping else f"{lemma}2"] = idx
    return mapping


class ImageNetFolderDataset(Dataset):
    """
    PyTorch Dataset for ImageNet stored as raw image folders.

    Args:
        data_dir: Root containing the split directories (e.g. `train/`, `validation/`).
        split: "train" or "val".
        transform: Optional transform applied to the PIL image.
        condition_type: "label" (int) or "text" (string prompts).
        prompt_template: Template for text prompts.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[object] = None,
        condition_type: str = "label",
        prompt_template: str = "a photo of a {class_name}",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.condition_type = condition_type
        self.prompt_template = prompt_template

        candidates = [self.data_dir / d for d in SPLIT_DIRS.get(split, (split,))]
        split_dir = next((p for p in candidates if p.is_dir()), None)
        if split_dir is None:
            raise FileNotFoundError(f"Split '{split}' not found; tried {[str(p) for p in candidates]}")
        self.split_dir = split_dir

        class_dirs = sorted(e.name for e in os.scandir(split_dir) if e.is_dir())
        name_to_label = _class_dir_to_label()
        unknown = [c for c in class_dirs if c not in name_to_label]
        if unknown:
            raise ValueError(f"Unrecognized ImageNet class dirs in {split_dir}: {unknown[:5]}")

        # Compact storage: numpy bytes arrays pickle cheaply into spawned workers.
        paths, labels = [], []
        for class_dir in class_dirs:
            label = name_to_label[class_dir]
            files = sorted(
                e.name for e in os.scandir(split_dir / class_dir)
                if e.is_file() and e.name.lower().endswith(IMG_EXTENSIONS)
            )
            paths.extend(os.path.join(class_dir, f) for f in files)
            labels.extend([label] * len(files))
        self.paths = np.array(paths, dtype=np.bytes_)
        self.labels = np.array(labels, dtype=np.int16)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, str]]:
        path = self.split_dir / self.paths[idx].decode()
        label = int(self.labels[idx])
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.condition_type == "text":
            return image, self.prompt_template.format(class_name=IMAGENET_CLASSES[label])
        return image, label

    @property
    def num_classes(self) -> int:
        return 1000
