import torch
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
    
def random_reveal(board: GameEnvironment, num_reveals=10):
    """Randomly reveals safe and still-hidden cells on the board. Never reveals mines."""
    if num_reveals is None:
        num_reveals = random.randint(3, 10)

    attempts = 0
    max_attempts = num_reveals * 10
    revealed = set()

    while len(revealed) < num_reveals and attempts < max_attempts:
        attempts += 1

        x = random.randint(0, board.size - 1)
        y = random.randint(0, board.size - 1)

        # Skip if already revealed
        if board.mask_board[x, y] != board.HIDDEN:
            continue

        # Skip if it's a mine
        if board.board[x, y] == board.MINE:
            continue

        board.reveal((x, y))
        revealed.add((x, y))

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
    Creates a label grid (H, W) with:
      1 = hidden safe cell
      0 = hidden mine cell
      -1 = revealed cell (ignore in loss)
    """
    H, W = board.size, board.size

    labels = torch.full((H, W), -1.0)  # default ignore

    mask = board.mask_board
    true_board = board.board

    hidden = (mask == board.HIDDEN)
    mines  = (true_board == board.MINE)

    # Hidden mines = 0
    labels[hidden & mines] = 0.0

    # Hidden safe = 1
    labels[hidden & (~mines)] = 1.0

    return labels


def main():
    board = generate_board(difficulty='hard')
    board = random_reveal(board, num_reveals=5)
    board.visualize_board(board.mask_board)

    encoded_mask = encode_mask_board(board)
    print("Encoded mask shape:", encoded_mask.shape)

    labels = make_label_tensor(board)
    print("Label tensor shape:", labels.shape)

if __name__ == "__main__":
    main()

# Run with:
# $ python -m Minesweeper.dataset.board_generation_utils