import torch
import torch.nn.functional as F
import random

from Minesweeper.GameEnvironment import GameEnvironment

def generate_board(size=22, difficulty='medium'):
    """Generates a Minesweeper board with a random starting point revealed."""
    board = GameEnvironment(size=size, difficulty=difficulty)
    start_point = (random.randint(0, size-1), random.randint(0, size-1))
    board.place_mines(start_point)
    board.compute_clue_values()
    board.reveal(start_point)
    return board
    
def game_reveal(board: GameEnvironment, max_moves=20):
    """
    Simulates a player revealing a connected path of safe cells. Makes the frontier boundary. 
    """

    # 'max_moves' steps of expanding the frontier (hidden cells adjacent to revealed cells)
    for _ in range(max_moves):
        if board.won_game():
            break

        candidates = []
        
        # Check neighbors of all currently revealed cells
        rows, cols = (board.mask_board != board.HIDDEN).nonzero(as_tuple=True)
        
        for r, c in zip(rows, cols):
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: 
                        continue
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < board.size and 0 <= nc < board.size:
                        # Must be HIDDEN and SAFE
                        if board.mask_board[nr, nc] == board.HIDDEN:
                            if board.board[nr, nc] != board.MINE:
                                candidates.append((nr, nc))

        if not candidates:
            break # Board is solved or stuck

        # Pick a random safe frontier cell to reveal
        move = random.choice(candidates)
        board.reveal(move)

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
      11: known mine
    """

    mask = board.mask_board.long()

    H, W = mask.shape
    C = 12
    encoding = torch.zeros((C, H, W), dtype=torch.float32)

    # Hidden cells
    encoding[0][mask == board.HIDDEN] = 1.0

    # Revealed numeric cells (0 to 8)
    for n in range(9):
        encoding[n + 1][mask == n] = 1.0

    # Revealed mine (not for Task 1)
    encoding[10][mask == board.MINE] = 1.0

    # Known mine (not for Task 1)
    encoding[11][mask == -5] = 1.0  # Change later

    return encoding

def make_label_tensor(board: GameEnvironment):
    """
    Creates a label grid (H, W).
    1 = hidden safe cell ADJACENT to a clue
    0 = hidden mine cell ADJACENT to a clue
    -1 = ignore (revealed cells OR hidden cells far from clues)
    """
    H, W = board.size, board.size
    labels = torch.full((H, W), -1.0) # Default to ignore

    mask = board.mask_board
    true_board = board.board
    
    # Identify Revealed Clue Cells (0-8), want to find neighbors of these cells with simple convolution to dilate the revealed area
    revealed_clues = (mask >= 0) & (mask <= 8)
    
    if not revealed_clues.any():
        return labels # Empty board, nothing to predict

    # Convert to float for convolution
    revealed_float = revealed_clues.float().unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    
    # Create a 3x3 kernel of ones to find all neighbors
    kernel = torch.ones((1, 1, 3, 3))
    
    # Convolve to find neighbors (padding=1 preserves size)
    #   --> creates a map where any cell touching a revealed clue becomes > 0
    frontier_map = F.conv2d(revealed_float, kernel, padding=1)[0, 0] > 0
    
    # Find Target Cells: Hidden cells on the frontier
    hidden = (mask == board.HIDDEN)
    targets = hidden & frontier_map
    
    # Assign Labels only to Valid Targets
    mines = (true_board == board.MINE)
    
    labels[targets & mines] = 0.0
    labels[targets & (~mines)] = 1.0
    
    return labels


def main():
    board = generate_board(difficulty='hard')
    board = game_reveal(board, num_reveals=5)
    board.visualize_board(board.mask_board)

    encoded_mask = encode_mask_board(board)
    print("Encoded mask shape:", encoded_mask.shape)

    labels = make_label_tensor(board)
    print("Label tensor shape:", labels.shape)

if __name__ == "__main__":
    main()

# Run with:
# $ python -m Minesweeper.dataset.board_generation_utils