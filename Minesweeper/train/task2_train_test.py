import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from Minesweeper.train.train_utils import save_checkpoint, load_checkpoint

def train_model(model: nn.Module, train_loader, test_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=False, device="cpu", 
                checkpoint_path=None, resume=False, save_every=1, strict_load=True):
    """
    Train with optional checkpointing.          
    """
    model.to(device)

    # --- Setup ---
    criterion = nn.MSELoss()
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

        for batch_idx, (inputs, move_masks, true_steps) in enumerate(train_prog, start=1):
            inputs = inputs.to(device, non_blocking=True)      # (B, 1, H, W)
            move_masks = move_masks.to(device, non_blocking=True) # (B, 1, H, W)
            true_steps = true_steps.to(device, dtype=torch.float32, non_blocking=True) # (B,)

            predictions_map = model(inputs) # (B, 1, H, W)

            # Only need value for the move taken
            predicted_value = (predictions_map * move_masks).sum(dim=(1, 2, 3)) 

            loss = criterion(predicted_value, true_steps)
            
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

        with torch.no_grad(): 
            for inputs, move_masks, true_steps in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                move_masks = move_masks.to(device, non_blocking=True)
                true_steps = true_steps.to(device, dtype=torch.float32, non_blocking=True)

                predictions_map = model(inputs) 
                predicted_value = (predictions_map * move_masks).sum(dim=(1, 2, 3)) 
                
                loss = criterion(predicted_value, true_steps)
                val_running_loss += loss.item() * inputs.size(0)

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
        plt.savefig(f'Minesweeper/plots/critic_model_v0_loss_plot.png')
        plt.show()

    return model, train_losses, test_losses


def test_model(model: nn.Module, test_loader, device="cpu"):
    """
    Evaluates the model on the test set for regression (Step Prediction).
    Returns average MSE Loss.
    """
    model.to(device)
    model.eval()
    
    # Use MSE for evaluation metric
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, move_masks, true_steps in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            move_masks = move_masks.to(device, non_blocking=True)
            true_steps = true_steps.to(device, dtype=torch.float32, non_blocking=True)

            predictions_map = model(inputs)
            predicted_value = (predictions_map * move_masks).sum(dim=(1, 2, 3))

            loss = criterion(predicted_value, true_steps)
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss