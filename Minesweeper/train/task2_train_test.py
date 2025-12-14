import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from Minesweeper.train.train_utils import save_checkpoint, load_checkpoint

def train_model(model: nn.Module, difficulty, train_loader, test_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=False, device="cpu", 
                checkpoint_path=None, resume=False, save_every=1, strict_load=True):
    """
    Train with optional checkpointing.          
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
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)

            valid = (labels != -1.0)
            targets = torch.clamp(labels, 0.0, 1.0)

            loss_map = criterion(logits, targets)

            # Explicitly weight the 'Mine' class (0.0) higher to force learning.
            
            # Create a weight tensor (default 1.0)
            weights = torch.ones_like(targets)
            
            # Assign higher weight to Mines (Label 0.0) -> 5.0 since safe are 5x more common
            weight_dict = {
                'easy': 10.0, 
                'medium': 5.0,
            }
            weights[targets == 0.0] = weight_dict.get(difficulty) 
            
            # Apply weights to the loss map
            loss_map = loss_map * weights

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
                labels = labels.to(device, non_blocking=True)

                logits = model(inputs)
                valid = (labels != -1.0)
                targets = torch.clamp(labels, 0.0, 1.0)

                loss_map = criterion(logits, targets)
                weights = torch.ones_like(targets)
                weights[targets == 0.0] = 5.0 
                loss_map = loss_map * weights
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
        plt.savefig(f'Minesweeper/plots/{difficulty}_loss_plot.png')
        plt.show()

    return model, train_losses, test_losses


def test_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            valid = (labels != -1.0)
            targets = torch.clamp(labels, 0.0, 1.0)

            preds = (torch.sigmoid(logits) >= 0.5)
            correct += ((preds == (targets >= 0.5)) & valid).sum().item()
            total += valid.sum().item()

    precision = correct / max(total, 1)
    return precision