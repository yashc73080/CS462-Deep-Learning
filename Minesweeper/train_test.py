import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



def _atomic_torch_save(obj, path) -> None:
    """Write to a temp file then replace to reduce chance of corrupted checkpoints."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, train_losses=None, test_losses=None, extra=None):
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

def load_checkpoint(path, model, device="cpu", optimizer=None, scheduler=None, strict=True):
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

def _ensure_BCHW_labels(labels: torch.Tensor):
    """
    Task1Dataset labels are (B,H,W) floats with values {1,0,-1}.
    Convert to (B,1,H,W) float.
    """
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)
    return labels.float()

def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=False, device="cpu", 
                checkpoint_path=None, resume=False, save_every=1, strict_load=True):
    """
    Train with optional checkpointing.
    num_epochs: number of epochs to run in THIS call. If resuming, training continues from the checkpoint's epoch.

    Typical usage:
      - Fresh training + checkpoints:
          train_model(..., checkpoint_path="checkpoints/ckpt.pt", resume=False)

      - Resume after interruption or continue training further:
          train_model(..., num_epochs=20, checkpoint_path="checkpoints/ckpt.pt", resume=True)          
    """
    model.to(device)

    # --- Setup ---
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, test_losses = [], []

    # --- Save/Load ---

    # Resume if requested
    start_epoch = 0
    if resume:
        if not checkpoint_path:
            raise ValueError("resume=True requires checkpoint_path to be set.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        start_epoch, train_losses, test_losses, _ = load_checkpoint(checkpoint_path, model=model, device=device, optimizer=optimizer, 
                                                                    scheduler=scheduler, strict=strict_load)
        print(f"Resuming from checkpoint '{checkpoint_path}' at epoch {start_epoch}.")

    if checkpoint_path:
        os.makedirs("Minesweeper/checkpoints", exist_ok=True)
        print(f"[checkpoint] path = {checkpoint_path} (save_every={save_every} epochs)")

    def _optionally_save(epoch_idx_in_call, absolute_epoch):
        if not checkpoint_path:
            return
        if save_every is None or save_every <= 0:
            return
        if (epoch_idx_in_call + 1) % save_every == 0:
            # Save every `save_every` epochs
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, 
                            epoch=absolute_epoch, train_losses=train_losses, test_losses=test_losses)

    # --- Training ---

    for e in range(num_epochs):
        absolute_epoch = start_epoch + e

        model.train()
        running_loss = 0.0

        train_prog = tqdm(train_loader, desc=f'Train Epoch {absolute_epoch + 1}', leave=False, dynamic_ncols=True)

        for batch_idx, (inputs, labels) in enumerate(train_prog, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = _ensure_BCHW_labels(labels.to(device, non_blocking=True))

            logits = model(inputs)

            valid = (labels != -1.0)
            targets = torch.clamp(labels, 0.0, 1.0)

            loss_map = criterion(logits, targets)
            loss = (loss_map * valid).sum() / valid.sum().clamp_min(1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            train_prog.set_postfix(loss=float(loss.item()))

        scheduler.step()
        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_epoch_loss)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = _ensure_BCHW_labels(labels.to(device, non_blocking=True))

                logits = model(inputs)
                valid = (labels != -1.0)
                targets = torch.clamp(labels, 0.0, 1.0)

                loss_map = criterion(logits, targets)
                loss = (loss_map * valid).sum() / valid.sum().clamp_min(1)
                val_running_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(logits) >= 0.5)
                correct += ((preds == (targets >= 0.5)) & valid).sum().item()
                total += valid.sum().item()

        val_epoch_loss = val_running_loss / len(test_loader.dataset)
        test_losses.append(val_epoch_loss)

        print(f"Epoch {absolute_epoch + 1}: Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        _optionally_save(e, absolute_epoch)

    # Final save (always) to continue later even if save_every is large
    if checkpoint_path:
        last_epoch = start_epoch + num_epochs - 1
        save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, 
                        epoch=last_epoch, train_losses=train_losses, test_losses=test_losses, extra={"finished": True})

    # --- Plotting ---

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss")
        plt.legend()

        if checkpoint_path:
            base_name = os.path.basename(checkpoint_path).replace('.py', '')
            plot_path = f'Minesweeper/loss/{base_name}_loss_plot.png'
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)

        plt.show()

    return model, train_losses, test_losses


def test_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = _ensure_BCHW_labels(labels.to(device, non_blocking=True))

            logits = model(inputs)
            valid = (labels != -1.0)
            targets = torch.clamp(labels, 0.0, 1.0)

            preds = (torch.sigmoid(logits) >= 0.5)
            correct += ((preds == (targets >= 0.5)) & valid).sum().item()
            total += valid.sum().item()

    return correct / max(total, 1)