import os 
import torch
import torch.nn as nn
import torch.optim as optim

def _atomic_torch_save(obj, path) -> None:
    """Write to a temp file then replace to reduce chance of corrupted checkpoints."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def save_checkpoint(path, model: nn.Module, optimizer: optim.Optimizer = None, scheduler: optim.lr_scheduler._LRScheduler = None, epoch=0, train_losses=None, test_losses=None, extra=None):
    """Saves a full training checkpoint to resume training."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_losses": list(train_losses) if train_losses is not None else [],
        "test_losses": list(test_losses) if test_losses is not None else [],
        "extra": extra if extra is not None else {},
    }
    _atomic_torch_save(checkpt, path)

def load_checkpoint(path, model: nn.Module, device="cpu", optimizer: optim.Optimizer = None, scheduler: optim.lr_scheduler._LRScheduler = None, strict=True):
    """
    Loads a checkpoint and restores model (+ optimizer/scheduler if provided).
    Returns: (start_epoch, train_losses, test_losses, extra)
      - start_epoch is the next epoch index to run (ckpt_epoch + 1)
    """
    checkpt = torch.load(path, map_location=device)

    model.load_state_dict(checkpt["model_state_dict"], strict=strict)
    model.to(device)

    if optimizer is not None and checkpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpt["optimizer_state_dict"])

    if scheduler is not None and checkpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpt["scheduler_state_dict"])

    last_epoch = int(checkpt.get("epoch", -1))
    start_epoch = last_epoch + 1

    train_losses = checkpt.get("train_losses", []) or []
    test_losses = checkpt.get("test_losses", []) or []
    extra = checkpt.get("extra", {}) or {}

    return start_epoch, train_losses, test_losses, extra