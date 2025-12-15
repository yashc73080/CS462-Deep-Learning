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
    random.seed(os.getpid())
    np.random.seed(os.getpid() % 123456789)

    difficulty, bot_maker = args 

    # Initialize LogicBot
    if bot_maker:
        bot = bot_maker()
    else:
        bot = LogicBot(difficulty=difficulty)

    # Handle immediate game over
    if bot.game_over:
        board_input = bg_utils.encode_mask_board(bot.game_environment)
        return board_input, torch.zeros((bot.size, bot.size)).unsqueeze(0), 0.0

    # Fast Forward random amount
    moves_made = 0
    max_moves = random.randint(0, (bot.size * bot.size) // 4)

    while not bot.game_over and moves_made < max_moves:
        move = get_bot_move(bot) 
        if move is None:
            break

        step_bot(bot, move) 
        if bot.game_environment.lost:
            break

        moves_made += 1

    mask_board = bot.game_environment.mask_board
    is_hidden = (mask_board == bot.game_environment.HIDDEN)
    is_revealed = (mask_board >= 0).float()
    
    # 3x3 kernel to count revealed neighbors
    kernel = torch.ones((1, 1, 3, 3), device=mask_board.device)
    neighbor_counts = F.conv2d(is_revealed.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    
    is_frontier = is_hidden & (neighbor_counts.squeeze() > 0)
    frontier_moves = torch.nonzero(is_frontier, as_tuple=False).tolist()
    all_possible_moves = torch.nonzero(is_hidden, as_tuple=False).tolist()

    if not all_possible_moves or bot.game_environment.lost:
        # Game ended during fast-forward
        current_board = bg_utils.encode_mask_board(bot.game_environment)
        return current_board, torch.zeros((bot.size, bot.size)).unsqueeze(0), 0.0

    # 80% chance to pick from frontier if available, otherwise random
    if frontier_moves and random.random() < 0.8:
        target_move = random.choice(frontier_moves)
    else:
        target_move = random.choice(all_possible_moves)
    
    # Convert to tuple for indexing
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

        # Finish the game manually
        while not bot.game_environment.won_game() and not bot.game_environment.lost:
            move = get_bot_move(bot) 
            if move is None:
                break

            step_bot(bot, move) 
            if bot.game_environment.lost:
                break

            survival_steps += 1
    
    # Normalize the target to help training stability (dividing by 100)
    return current_board, move_mask.unsqueeze(0), float(survival_steps) / 100.0


class Task2Dataset(Dataset):
    def __init__(self, num_samples, difficulty='medium', cache_file="task2_data.pt", 
                 force_regenerate=False, bot_maker=None, num_workers=0):
        """
        num_workers: 
          0 = Run sequentially in main process (Safe for CUDA/GPU)
          >0 = Run with multiprocessing (Faster for LogicBot)
        """
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.bot_maker = bot_maker
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
        print(f"Generating {self.num_samples} samples. (Workers: {self.num_workers})")
        
        worker_args = [(self.difficulty, self.bot_maker) for _ in range(self.num_samples)]
        self.data = []

        if self.num_workers > 0:
            # MULTIPROCESSING
            with mp.Pool(processes=self.num_workers) as pool:
                for result in tqdm(pool.imap_unordered(simulate_worker, worker_args), total=self.num_samples, desc="Simulating Games"):
                    self.data.append(result)
        else:
            # SEQUENTIAL
            for args in tqdm(worker_args, desc="Simulating Games (Seq)"):
                self.data.append(simulate_worker(args))
        
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