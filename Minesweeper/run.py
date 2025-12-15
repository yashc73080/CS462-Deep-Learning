import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from Minesweeper.models.MinePredictionNet import MinePredictionNet
from Minesweeper.models.ActorCriticNet import CriticNet
from Minesweeper.train.train_utils import load_checkpoint
from Minesweeper.train.task2_train_test import test_model
from Minesweeper.dataset.Task1Dataset import Task1Dataset
from Minesweeper.dataset.Task2Dataset import Task2Dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    val_dataset = Task2Dataset(difficulty='easy', num_samples=2000, cache_file="val_easy_v0.pt")
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True)

    model = CriticNet(input_size=(12, 22, 22), device=device)

    ckpt_path = "Minesweeper/checkpoints/critic_model_v0.pth"
    start_epoch, train_losses, val_losses, extra = load_checkpoint(ckpt_path, model=model, device=device)

    final_train_loss = train_losses[-1] if len(train_losses) > 0 else None
    final_val_loss = val_losses[-1] if len(val_losses) > 0 else None

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint epoch (next start): {start_epoch}")
    print(f"Final train loss in checkpoint: {final_train_loss}")
    print(f"Final val loss in checkpoint:   {final_val_loss}")

    accuracy = test_model(model, val_loader, device=device)
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()