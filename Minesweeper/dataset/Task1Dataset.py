from torch.utils.data import DataLoader, Dataset
import random

from Minesweeper.dataset import board_generation_utils as bg_utils

class Task1Dataset(Dataset):
    def __init__(self, num_samples, difficulty='easy'):
        self.num_samples = num_samples
        self.difficulty = difficulty

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get a board with mines, clues, and random starting point 
        board = bg_utils.generate_board(difficulty=self.difficulty)

        # Randomly reveal some safe cells
        num_reveals = random.randint(3, 10)
        board = bg_utils.random_reveal(board, num_reveals=num_reveals)

        input_tensor = bg_utils.encode_mask_board(board)
        label_tensor = bg_utils.make_label_tensor(board)

        return input_tensor, label_tensor