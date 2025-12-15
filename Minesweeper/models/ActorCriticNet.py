# Task 2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
import numpy as np
import random
import os

from Minesweeper.GameEnvironment import GameEnvironment
from Minesweeper.train.train_utils import load_checkpoint

class CriticNet(nn.Module):
    def __init__(self, input_size=(1,22,22), hidden_dim=64, device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(input_size[0], hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        
        self.conv5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(hidden_dim)

        self.dropout = nn.Dropout(0.3)

        self.last_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x) 

        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Output
        x = self.last_conv(x)
        return x

class NeuralBot:
    def __init__(self, model_path, difficulty='medium', device='cpu'):
        self.device = device
        self.model = CriticNet(input_size=(1,22,22), device=device)
        
        self.model.eval()
        self.difficulty = difficulty
        

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

    difficulty = 'medium'

    # Load data
    train_dataset = Task2Dataset(difficulty=difficulty, num_samples=50000, cache_file=f"Minesweeper/dataset/train_{difficulty}_v0.pt")
    val_dataset = Task2Dataset(difficulty=difficulty, num_samples=5000, cache_file=f"Minesweeper/dataset/val_{difficulty}_v0.pt")

    num_workers = 0 # os.cpu_count() - 4
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)

    # Build and train model
    print(f'Training CriticNet on {difficulty} difficulty...')
    model = CriticNet(input_size=(1,22,22), device=device)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=True, device=device,
                                                  checkpoint_path=f'Minesweeper/checkpoints/critic_model_{difficulty}_v0.pth', save_every=1, resume=False)
    accuracy = test_model(model, val_loader, device=device)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()
