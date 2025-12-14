from torch.utils.data import Dataset
import random

from Minesweeper.dataset import board_generation_utils as bg_utils
from Minesweeper.LogicBot import LogicBot

class Task2Dataset(Dataset):
    def __init__(self, num_samples, difficulty='medium'):
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.logic_bot = LogicBot(difficulty=difficulty)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        board = bg_utils.generate_board(difficulty=self.difficulty)