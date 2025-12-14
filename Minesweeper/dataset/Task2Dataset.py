import torch
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm 

from Minesweeper.dataset import board_generation_utils as bg_utils
from Minesweeper.LogicBot import LogicBot

class Task2Dataset(Dataset):
    def __init__(self, num_samples, difficulty='medium', cache_file="task2_data.pt", force_regenerate=False):
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.data = []

        # Check if we can load from disk to save time
        if os.path.exists(cache_file) and not force_regenerate:
            print(f"Loading pre-generated dataset from {cache_file}...")
            loaded_data = torch.load(cache_file)
            if len(loaded_data) == num_samples:
                self.data = loaded_data
            else:
                print(f"Cached dataset size ({len(loaded_data)}) matches request ({num_samples}). Regenerating...")
                self.generate_dataset(cache_file)
        else:
            self.generate_dataset(cache_file)

    def generate_dataset(self, cache_file):
        print(f"Generating {self.num_samples} samples.")
        
        self.data = []
        for _ in tqdm(range(self.num_samples), desc="Simulating Games"):
            self.data.append(self.simulate_one_game())
        
        print(f"Saving dataset to {cache_file}...")
        torch.save(self.data, cache_file)

    def simulate_one_game(self):
        # Initialize LogicBot
        bot = LogicBot(difficulty=self.difficulty)
        
        # Setup Board
        bot.game_environment = bg_utils.generate_board(difficulty=self.difficulty)
        bot.update_after_reveal()
        bot.run_inference()

        # Handle immediate game over
        if bot.game_over:
             board_input = bot.game_environment.mask_board.clone().detach().float().unsqueeze(0)
             return board_input, torch.zeros((bot.size, bot.size)).unsqueeze(0), 0.0

        # Fast Forward random amount
        moves_made = 0
        max_moves = random.randint(0, (bot.size * bot.size) // 4)

        while not bot.game_over and moves_made < max_moves:
            if len(bot.inferred_safe) > 0:
                move = bot.inferred_safe.pop()
            else:
                possible = bot.cells_remaining - bot.inferred_mine
                if not possible: break
                move = random.choice(list(possible))

            bot.game_environment.reveal(move)
            if bot.game_environment.lost:
                break
            
            bot.update_after_reveal()
            bot.run_inference()
            moves_made += 1

        # Intervention
        possible_moves = list(bot.cells_remaining)
        if not possible_moves or bot.game_over:
             board_input = bot.game_environment.mask_board.clone().detach().float().unsqueeze(0)
             return board_input, torch.zeros((bot.size, bot.size)).unsqueeze(0), 0.0

        target_move = random.choice(possible_moves)
        
        # Save Inputs
        current_board = bot.game_environment.mask_board.clone().detach().float().unsqueeze(0)
        move_mask = torch.zeros((bot.size, bot.size), dtype=torch.float32)
        move_mask[target_move] = 1.0
        
        # Evaluate Survival
        bot.game_environment.reveal(target_move)
        
        survival_steps = 0
        if not bot.game_environment.lost:
            survival_steps += 1
            bot.update_after_reveal()
            bot.run_inference()

            # Finish the game manually
            while not bot.game_over:
                if len(bot.inferred_safe) > 0:
                    move = bot.inferred_safe.pop()
                else:
                    possible = bot.cells_remaining - bot.inferred_mine
                    if not possible: break
                    move = random.choice(list(possible))

                bot.game_environment.reveal(move)
                if bot.game_environment.lost:
                    break
                
                survival_steps += 1
                bot.update_after_reveal()
                bot.run_inference()
                if bot.game_environment.won_game():
                    break
        
        # Normalize the target to help training stability (dividing by 100)
        return current_board, move_mask.unsqueeze(0), float(survival_steps) / 100.0

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]