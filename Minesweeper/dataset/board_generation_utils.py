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
