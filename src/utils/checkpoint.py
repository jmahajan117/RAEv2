"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def save_stage1_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
    keep_last: Optional[int] = None,
) -> None:
    """Save Stage 1 training checkpoint (model + discriminator).

    keep_last: if set, delete all but the newest `keep_last` ep-*.pt in the same dir after
    the new checkpoint is fully written (None keeps everything).
    """
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write-then-rename: a job killed mid-save leaves only a .tmp (ignored by resume), never a
    # truncated .pt, so the newest .pt is always loadable.
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    if keep_last:
        prune_checkpoints(os.path.dirname(path), keep_last)


def prune_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
    """Keep only the newest `keep_last` ep-*.pt checkpoints (by epoch) and drop stale .tmp files."""
    ckpts = []
    for f in glob.glob(os.path.join(checkpoint_dir, "ep-*.pt")):
        match = re.search(r"ep-(\d+)\.pt$", f)
        if match:
            ckpts.append((int(match.group(1)), f))
    for _, f in sorted(ckpts)[:-keep_last]:
        os.remove(f)
    for f in glob.glob(os.path.join(checkpoint_dir, "*.pt.tmp")):
        os.remove(f)


def load_stage1_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 1 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    keep_last: Optional[int] = None,
) -> None:
    """Save Stage 2 training checkpoint.

    keep_last: if set, delete all but the newest `keep_last` ep-*.pt in the same dir after
    the new checkpoint is fully written (None keeps everything).
    """
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write-then-rename (see save_stage1_checkpoint): the newest .pt is always loadable.
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    if keep_last:
        prune_checkpoints(os.path.dirname(path), keep_last)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
]
