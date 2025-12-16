import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from time import time

from Minesweeper.GameEnvironment import GameEnvironment
import Minesweeper.dataset.board_generation_utils as bg_utils


class ThinkingBlock(nn.Module):
    def __init__(self, channels, device='cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.to(device)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + identity)
    
class ThinkingNet(nn.Module):
    def __init__(self, input_size=(12,22,22), hidden_channels=64, max_steps=10, device='cpu'):
        super().__init__()
        self.max_steps = max_steps

        # Preprocess board into feature space
        self.encoding = nn.Conv2d(input_size[0], hidden_channels, kernel_size=3, padding=1)

        # Thinking block (reuse)
        self.thought = ThinkingBlock(hidden_channels, device=device)

        # Prediction
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        self.to(device)

    def forward(self, x, steps=None):
        if steps is None:
            steps = self.max_steps

        # Thought state
        h = self.encoding(x)

        outputs = []

        # Thinking process
        for _ in range(steps):
            h = self.thought(h)

            pred = self.final_conv(h)
            outputs.append(pred)

        return torch.stack(outputs, dim=1)
    
@torch.no_grad()
def thinking_policy(model, board: GameEnvironment, device="cpu", steps=10):
    """
    Runs the ThinkingNet for `steps` iterations and picks the move based on the FINAL thought (index -1).
    """
    model.eval()

    encoded = bg_utils.encode_mask_board(board).unsqueeze(0).to(device)
    
    # Forward pass: Returns [Batch, Steps, 1, H, W]
    outputs_over_time = model(encoded, steps=steps)
    
    # Get final step prediction: [1, H, W]
    final_logits = outputs_over_time[0, -1] 
    probs = torch.sigmoid(final_logits)[0] # [H, W]

    mask = board.mask_board
    hidden = (mask == board.HIDDEN)

    if not hidden.any():
        return None

    safe_probs = probs.clone()
    safe_probs[~hidden] = -float('inf')

    # Find the single safest cell on the board
    idx = torch.argmax(safe_probs)
    x = idx // board.size
    y = idx % board.size
    return int(x), int(y)

# --- 2. Copy Game Loop to use the new Policy ---
def play_one_game_thinking(model, device="cpu", difficulty="medium", size=22, steps=10):
    board = GameEnvironment(size=size, difficulty=difficulty)

    # Standard start
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

        move = thinking_policy(model, board, device=device, steps=steps)

        if move is None:
            # Fallback 
            hidden_cells = [(x, y) for x in range(size) for y in range(size) 
                            if board.mask_board[x, y] == board.HIDDEN]
            if not hidden_cells: 
                return False, safe_moves, mines_triggered
            move = random.choice(hidden_cells)

        success = board.reveal(move)
        if success:
            safe_moves += 1
    

def main():
    from Minesweeper.dataset.Task1Dataset import Task1Dataset
    from Minesweeper.train.task3_train_test import train_model, test_model

    start_time = time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    difficulty = 'medium'

    # Load data
    train_dataset = Task1Dataset(num_samples=15000, difficulty=difficulty)
    val_dataset = Task1Dataset(num_samples=1000, difficulty=difficulty)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True)

    # Build and train model
    print(f'Training ThinkingNet on {difficulty} difficulty...')
    model = ThinkingNet(input_size=(12,22,22), hidden_channels=64, max_steps=10, device=device)
    model, train_losses, val_losses = train_model(model, difficulty, train_loader, val_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=True, device=device,
                                                  checkpoint_path=f'Minesweeper/checkpoints/thinking_{difficulty}_model.pth', save_every=1, resume=False)
    precision = test_model(model, val_loader, device=device)

    print(f'Final Validation Precision: {precision:.4f}')

    end_time = time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()