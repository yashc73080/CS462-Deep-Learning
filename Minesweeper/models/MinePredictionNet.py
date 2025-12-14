# Task 1

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from time import time

from Minesweeper.GameEnvironment import GameEnvironment
from Minesweeper.dataset.board_generation_utils import encode_mask_board

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, device='cpu'):
        super().__init__()

        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_channels)
            )

        self.to(device)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Add skip connection
        x = self.relu(x + identity)
        return x
    
class MinePredictionNet(nn.Module):
    def __init__(self, input_size=(12, 22, 22), device='cpu'):
        super().__init__()

        # Initial convolutional layer
        self.conv = nn.Conv2d(input_size[0], 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16, dilation=1, device=device),
            ResidualBlock(16, 16, dilation=1, device=device),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, dilation=2, device=device),
            ResidualBlock(32, 32, dilation=2, device=device),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, dilation=4, device=device),
            ResidualBlock(64, 64, dilation=4, device=device),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64, dilation=8, device=device),
            ResidualBlock(64, 64, dilation=8, device=device),
        )

        self.last_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.to(device)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.last_conv(x)

        return x
    
@torch.no_grad()
def neural_policy(model, board: GameEnvironment, device="cpu"):
    """
    Choose the hidden cell with the highest predicted probability of being safe.
    """
    model.eval()

    encoded = encode_mask_board(board).unsqueeze(0).to(device)  # (1,C,H,W)
    logits = model(encoded)  # (1,1,H,W)
    probs = torch.sigmoid(logits)[0, 0]  # (H,W)

    mask = board.mask_board
    hidden = (mask == board.HIDDEN)

    if not hidden.any():
        return None

    # Mask out non-hidden cells
    probs = probs.clone()
    probs[~hidden] = -1.0

    # Pick safest cell
    idx = torch.argmax(probs)
    x = idx // board.size
    y = idx % board.size
    return int(x), int(y)

def play_one_game_nn(model, device="cpu", difficulty="medium", size=22):
    board = GameEnvironment(size=size, difficulty=difficulty)

    start = (random.randint(0, size - 1), random.randint(0, size - 1))
    board.place_mines(start)
    board.compute_clue_values()
    board.reveal(start)

    safe_moves = 0
    mines_triggered = 0

    while True:
        if board.won_game():
            return True, safe_moves, mines_triggered

        if board.lost:
            mines_triggered += 1
            return False, safe_moves, mines_triggered

        move = neural_policy(model, board, device)
        if move is None:
            return False, safe_moves, mines_triggered

        success = board.reveal(move)
        if success:
            safe_moves += 1


def main():
    from Minesweeper.dataset.Task1Dataset import Task1Dataset
    from Minesweeper.train_test import train_model, test_model

    start_time = time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data
    difficulty = 'medium'

    train_dataset = Task1Dataset(num_samples=30000, difficulty=difficulty)
    val_dataset = Task1Dataset(num_samples=3000, difficulty=difficulty)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Build and train model
    medium_model = MinePredictionNet(input_size=(12, 22, 22), device=device)
    medium_model, med_train_losses, med_val_losses = train_model(medium_model, train_loader, val_loader, num_epochs=10, lr=0.002, decay=0.0001, plot=True, device=device,
                                                                checkpoint_path='Minesweeper/checkpoints/medium_model.pth', resume=False)
    med_acc = test_model(medium_model, val_loader, device=device)

    print(f'Final Validation Accuracy: {med_acc:.4f}')

    end_time = time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()


# Run with: 
# $ python -m Minesweeper.models.MinePredictionNet