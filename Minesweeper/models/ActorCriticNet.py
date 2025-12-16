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
import Minesweeper.dataset.board_generation_utils as bg_utils

class CriticNet(nn.Module):
    def __init__(self, input_size=(12,22,22), device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(0.5) 

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Simplified Head
        self.last_conv = nn.Conv2d(32, 1, kernel_size=1)

        self.to(device)

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
        x = self.dropout(x)

        # Output
        x = self.last_conv(x)
        return x

class NeuralBot:
    def __init__(self, model_path, difficulty='medium', device='cpu'):
        self.device = device
        self.model = CriticNet(input_size=(12,22,22), device=device)
        load_checkpoint(model_path, model=self.model, device=device)
        self.model.eval()
        self.game_environment = GameEnvironment(difficulty=difficulty)
        self.size = self.game_environment.size
        self.game_over = False

    def get_best_move(self):
        board_tensor = bg_utils.encode_mask_board(self.game_environment).unsqueeze(0).to(self.device)

        # Critic prediction
        with torch.no_grad():
            output = self.model(board_tensor)
            prediction_map = output.squeeze().cpu().numpy()

        # Mask out already revealed cells
        mask = self.game_environment.mask_board.cpu().numpy()
        prediction_map[mask != self.game_environment.HIDDEN] = -float('inf')

        # Get the cell with the highest score
        best_move = divmod(prediction_map.argmax(), self.size)

        if prediction_map.max() < 0.01:
            print("WARNING: Model is predicting extremely low values (Model Collapse)")

        return best_move
    
    def solve(self):
        start_x = random.randint(0, self.size - 1)
        start_y = random.randint(0, self.size - 1)
        self.game_environment.place_mines((start_x, start_y))
        self.game_environment.compute_clue_values()
        self.game_environment.reveal((start_x, start_y))

        survived = 0
        while not self.game_environment.won_game() and not self.game_environment.lost:
            move = self.get_best_move()
            self.game_environment.reveal(move)
            if self.game_environment.lost:
                self.game_over = True
                break
            survived += 1

        return survived, self.game_environment.won_game()


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
    gen = 1

    # Actor = LogicBot, Critic = CriticNet
    if gen == 0:
        print('Generation 0 - Training CriticNet with LogicBot data')
        # Load data
        train_dataset = Task2Dataset(difficulty=difficulty, num_samples=10000, cache_file=f"Minesweeper/dataset/train_{difficulty}_v0.pt", num_workers=12)
        val_dataset = Task2Dataset(difficulty=difficulty, num_samples=5000, cache_file=f"Minesweeper/dataset/val_{difficulty}_v0.pt", num_workers=12)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

        # Build and train model
        print(f'Training CriticNet on {difficulty} difficulty...')
        model = CriticNet(input_size=(12,22,22), device=device)
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0005, decay=0.0001, plot=True, device=device,
                                                    checkpoint_path=f'Minesweeper/checkpoints/critic_model_{difficulty}_v0.pth', save_every=1, resume=False)
        accuracy = test_model(model, val_loader, device=device)
        print(f'Test Accuracy: {accuracy:.4f}')
    
    # Actor = NeuralBot (using CriticNet), Critic = CriticNet
    elif gen == 1:
        print('Generation 1 - Training CriticNet with NeuralBot data')
        # Load data
        new_bot = NeuralBot(model_path=f"Minesweeper/checkpoints/critic_model_{difficulty}_v0.pth", difficulty=difficulty, device=device)
        train_dataset = Task2Dataset(difficulty=difficulty, num_samples=10000, cache_file=f"Minesweeper/dataset/train_{difficulty}_v1.pt", neural_bot=new_bot)
        val_dataset = Task2Dataset(difficulty=difficulty, num_samples=5000, cache_file=f"Minesweeper/dataset/val_{difficulty}_v1.pt", neural_bot=new_bot)

        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)

        # Build and train model
        print(f'Retraining CriticNet on {difficulty} difficulty (v1)...')
        model = CriticNet(input_size=(12,22,22), device=device) # Load previous weights
        load_checkpoint(f'Minesweeper/checkpoints/critic_model_{difficulty}_v0.pth', model=model, device=device)

        model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, decay=0.0001, plot=True, device=device,
                                                    checkpoint_path=f'Minesweeper/checkpoints/critic_model_{difficulty}_v1.pth', save_every=1, resume=False)
        accuracy = test_model(model, val_loader, device=device)
        print(f'Test Accuracy: {accuracy:.4f}')

    end_time = time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
