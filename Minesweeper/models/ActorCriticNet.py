# Task 2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
import numpy as np
import random
import os

class CriticNet(nn.Module):
    def __init__(self, input_size=(1,22,22), hidden_dim=64, device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(input_size[0], hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.last_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.last_conv(x)

        return x
        

def main():
    from Minesweeper.dataset.Task2Dataset import Task2Dataset
    from Minesweeper.train.task2_train_test import train_model, test_model

    start_time = time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data
    train_dataset = Task2Dataset(difficulty='easy', num_samples=10000, cache_file="train_easy_v0.pt")
    val_dataset = Task2Dataset(difficulty='easy', num_samples=2000, cache_file="val_easy_v0.pt")

    num_workers = os.cpu_count() - 4
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Build and train model
    print(f'Training CriticNet on easy difficulty...')
    model = CriticNet(input_size=(1,22,22), device=device)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=True, device=device,
                                                  checkpoint_path=f'Minesweeper/checkpoints/critic_model_v0.pth', save_every=1, resume=False)
    accuracy = test_model(model, val_loader, device=device)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()
