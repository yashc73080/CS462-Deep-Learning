import torch
import torch.nn.functional as F
import random

from Minesweeper.GameEnvironment import GameEnvironment

# Maps -5->11, -2->0, -1->10, 0..8 -> 1..9
ENCODING_MAP = torch.tensor([11, 0, 0, 0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)

def generate_board(size=22, difficulty='medium'):
    """Generates a Minesweeper board with a random starting point revealed."""
    board = GameEnvironment(size=size, difficulty=difficulty)
    start_point = (random.randint(0, size-1), random.randint(0, size-1))
    board.place_mines(start_point)
    board.compute_clue_values()
    board.reveal(start_point)
    return board

def encode_mask_board(board: GameEnvironment):
    """
    Encodes mask_board into a 12-channel one-hot tensor (C, H, W).
    
    Channels:
      0: hidden
      1: revealed 0
      2: revealed 1
      ...
      9: revealed 8
      10: revealed mine
      11: known mine (flag)
    """

    mask = board.mask_board.long()
    device = mask.device
    
    mapping = ENCODING_MAP.to(device)

    # Lookup Indices (Shape: H, W)
    # Add +5 so that -5 becomes index 0
    indices = mapping[mask + 5]

    # One-Hot Encode (Shape: H, W, 12)
    one_hot = F.one_hot(indices, num_classes=12)

    # Permute to Channel-First and convert to Float (Shape: 12, H, W)
    return one_hot.permute(2, 0, 1).float()