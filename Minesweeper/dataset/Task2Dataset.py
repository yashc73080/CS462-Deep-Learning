import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import numpy as np 
import os
from tqdm import tqdm 

from Minesweeper.LogicBot import LogicBot
import Minesweeper.dataset.board_generation_utils as bg_utils

# --- HELPER FUNCTIONS ---

def get_bot_move(bot):
    '''Get bot's next move'''
    # NeuralBot
    if hasattr(bot, 'get_best_move'):
        return bot.get_best_move()
    
    # LogicBot
    if len(bot.inferred_safe) > 0:
        return bot.inferred_safe.pop()
    else:
        possible = bot.cells_remaining - bot.inferred_mine
        if not possible:
            return None
        return random.choice(list(possible))
        
def step_bot(bot, move):
    '''Make bot take one step'''
    bot.game_environment.reveal(move)

    # LogicBot updates
    if hasattr(bot, 'update_after_reveal'):
        bot.update_after_reveal()
    if hasattr(bot, 'run_inference'):
        bot.run_inference()

def simulate_worker(args):
    difficulty, neural_bot = args 

    if neural_bot:
        bot = neural_bot
        bot.game_environment = bg_utils.generate_board(size=bot.size, difficulty=difficulty)
        bot.game_over = False
    else:
        bot = LogicBot(difficulty=difficulty)
        start_x = random.randint(0, bot.size - 1)
        start_y = random.randint(0, bot.size - 1)
        bot.game_environment.place_mines((start_x, start_y))
        bot.game_environment.compute_clue_values()
        bot.game_environment.reveal((start_x, start_y))
        
        bot.update_after_reveal()
        bot.run_inference()

    if bot.game_environment.lost: 
        return None

    # Fast Forward 
    moves_made = 0
    max_moves = random.randint(5, 15) 

    while not bot.game_environment.lost and not bot.game_environment.won_game() and moves_made < max_moves:
        move = get_bot_move(bot) 
        if move is None:
            break

        step_bot(bot, move) 
        if bot.game_environment.lost:
            break

        moves_made += 1

    if bot.game_environment.lost or bot.game_environment.won_game():
        return None 
        
    mask_board = bot.game_environment.mask_board
    is_hidden = (mask_board == bot.game_environment.HIDDEN)
    is_revealed = (mask_board >= 0).float()
    
    kernel = torch.ones((1, 1, 3, 3), device=mask_board.device)
    neighbor_counts = F.conv2d(is_revealed.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    
    is_frontier = is_hidden & (neighbor_counts.squeeze() > 0)
    frontier_moves = torch.nonzero(is_frontier, as_tuple=False).tolist()
    all_possible_moves = torch.nonzero(is_hidden, as_tuple=False).tolist()

    if not all_possible_moves or bot.game_environment.lost:
        return None

    # Pick Intervention Move
    if frontier_moves and random.random() < 0.8:
        target_move = random.choice(frontier_moves)
    else:
        target_move = random.choice(all_possible_moves)
    
    target_move = tuple(target_move) 
    
    # Save Inputs
    current_board = bg_utils.encode_mask_board(bot.game_environment)
    move_mask = torch.zeros((bot.size, bot.size), dtype=torch.float32)
    move_mask[target_move] = 1.0
    
    # Evaluate Survival
    step_bot(bot, target_move) 
    
    survival_steps = 0
    if not bot.game_environment.lost:
        survival_steps += 1
        while not bot.game_environment.won_game() and not bot.game_environment.lost:
            move = get_bot_move(bot) 
            if move is None: break
            step_bot(bot, move) 
            if bot.game_environment.lost: break
            survival_steps += 1
    
    # Apply Safety Bonus (0.5) so model learns 0.0 vs 0.5+
    if survival_steps == 0:
        label_value = 0.0
    else:
        label_value = 0.5 + (float(survival_steps) / 200.0)

    return current_board, move_mask.unsqueeze(0), label_value

class Task2Dataset(Dataset):
    def __init__(self, num_samples, difficulty='medium', cache_file="task2_data.pt", 
                 force_regenerate=False, neural_bot=None, num_workers=0):
        """
        num_workers: 
          0 = Run sequentially in main process (Safe for CUDA/GPU)
          >0 = Run with multiprocessing (Faster for LogicBot)
        """
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.neural_bot = neural_bot
        self.num_workers = num_workers
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
        # Define attempts = num_samples * 5 because we discard ~75% of games 
        attempts = self.num_samples * 5
        print(f"Generating {self.num_samples} samples (Max Attempts: {attempts}, Workers: {self.num_workers})")
        
        worker_args = [(self.difficulty, self.neural_bot) for _ in range(attempts)]
        self.data = []

        if self.num_workers > 0:
            pool = mp.Pool(processes=self.num_workers)
            iterator = pool.imap_unordered(simulate_worker, worker_args, chunksize=10)
        else:
            iterator = (simulate_worker(arg) for arg in worker_args)
        
        avg_label = 0.0
        
        pbar = tqdm(total=self.num_samples, desc="Collecting Valid Games")
        
        for result in iterator:
            if result is not None:
                self.data.append(result)
                avg_label += result[2]
                pbar.update(1)
                
                if len(self.data) >= self.num_samples:
                    break
        
        pbar.close()
        
        if self.num_workers > 0:
            pool.close()
            pool.join()

        if len(self.data) == 0:
            raise RuntimeError("Generated 0 valid samples! Check if Bot is losing immediately.")
        
        avg_label /= len(self.data)
        print(f"Collected {len(self.data)} samples. Average Label (Normalized Steps): {avg_label:.4f}")
        
        if avg_label < 0.01:
            print("\nWARNING: Average label is extremely low. The bot might be dying immediately in every game.")
            print("Check if the inputs to the model are correct or if the difficulty is too high.\n")

        print(f"Saving dataset to {cache_file}...")
        torch.save(self.data, cache_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, move_mask, label = self.data[idx]

        # Data Augmentation
        if random.random() > 0.5:
            board = torch.flip(board, [2])
            move_mask = torch.flip(move_mask, [2])
        if random.random() > 0.5:
            board = torch.flip(board, [1])
            move_mask = torch.flip(move_mask, [1])
        k = random.randint(0, 3)
        board = torch.rot90(board, k, [1, 2])
        move_mask = torch.rot90(move_mask, k, [1, 2])

        return board, move_mask, label