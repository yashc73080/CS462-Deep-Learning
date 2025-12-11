import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque

class GameEnvironment():

    def __init__(self, size=22, difficulty='medium'):
        difficulty_map = {
            'easy': 50,
            'medium': 80,
            'hard': 100
        }

        self.size = size
        self.num_mines = difficulty_map.get(difficulty, 80)
        self.board = self.initialize_board(0) # The actual game board
        
        self.HIDDEN = -2 # any hidden cell
        self.mask_board = self.initialize_board(self.HIDDEN) # The board shown to the player

    def initialize_board(self, num):
        '''
        Initializes an empty game board. 0 represents an empty cell. -1 represents a mine.
        '''
        return torch.full((self.size, self.size), num, dtype=torch.int32)
    
    def place_mines(self, first_click):
        '''
        Places mines on the board, ensuring the first clicked cell is not a mine.
        '''
        x, y = first_click
        mines_placed = 0

        while mines_placed < self.num_mines:
            # Randomly select a cell
            mine_x = random.randint(0, self.size - 1)
            mine_y = random.randint(0, self.size - 1)

            # Place mine if it's not the first clicked cell and not already a mine
            if (mine_x, mine_y) != (x, y) and self.board[mine_x][mine_y] != -1:
                self.board[mine_x][mine_y] = -1
                mines_placed += 1

        return self.board
    
    def compute_clue_values(self):
        '''
        Computes the clue values for each cell in the board based on adjacent mines.
        '''
        directions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        
        def get_clue_value(position, board):
            # Helper function to count adjacent mines
            clue = 0
            x, y = position
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if board[nx][ny] == -1:
                        clue += 1
            return clue
        
        # Compute clue values for non-mine cells
        for x in range(self.size):
            for y in range(self.size):

                if self.board[x][y] != -1:
                    self.board[x][y] = get_clue_value((x, y), self.board)

        return self.board

    def reveal(self, clicked_cell):
        '''
        Reveals cells following Minesweeper rules:
        - Clicking a mine ends the game and reveals the full board.
        - Clicking a number reveals just that cell.
        - Clicking a 0 reveals the connected 0-region and its bordering numbers.

        Returns: True if safe click, False if mine.
        '''
        x, y = clicked_cell

        if not (0 <= x < self.size and 0 <= y < self.size):
            return True

        # Ignore already revealed cells
        if self.mask_board[x][y] != self.HIDDEN:
            return True

        value = int(self.board[x][y])

        if value == -1:
            # Game over: reveal full board
            self.mask_board = self.board.clone()
            print("Clicked on a mine! Game over.")
            return False

        if value > 0:
            # Reveal just this cell
            self.mask_board[x][y] = self.board[x][y]
            return True

        # value == 0
        self.flood_fill_reveal(clicked_cell)
        return True

    def flood_fill_reveal(self, start_cell):
        '''
        BFS flood fill:
        - Reveals all connected 0-cells reachable from start_cell
        - Also reveals bordering number cells (but does not expand them)
        '''
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

        q = deque([start_cell])

        while q:
            x, y = q.popleft()

            if not (0 <= x < self.size and 0 <= y < self.size):
                continue
            if self.mask_board[x][y] != self.HIDDEN:
                continue
            if int(self.board[x][y]) == -1:
                continue  # never reveal mines with flood fill

            # Reveal current cell
            self.mask_board[x][y] = self.board[x][y]

            # If 0, expand to neighbors
            if int(self.board[x][y]) == 0:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if self.mask_board[nx][ny] == self.HIDDEN and int(self.board[nx][ny]) != -1:
                            q.append((nx, ny))
    
    def visualize_board(self, board=None, title="Game Board", show=True):
        '''
        Mines are shown in red, clue values in blue gradient, empty cells in white.
        '''
        if board is None:
            board = self.board
        
        board_array = board.numpy().astype(float)
        
        # Replace mines with a high value for color mapping
        board_display = board_array.copy()
        board_display[board_display == -1] = self.num_mines + 1
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create heatmap
        sns.heatmap(board_display, cmap='YlOrRd', cbar=False, 
                    linewidths=0.5, linecolor='black', ax=ax,
                    vmin=0, vmax=self.num_mines + 1)
        
        # Add text annotations for clue values and mines
        for i in range(self.size):
            for j in range(self.size):
                if board_array[i][j] == -1:
                    ax.text(j + 0.5, i + 0.5, 'M', 
                        ha='center', va='center', fontsize=10)
                elif board_array[i][j] > 0:
                    ax.text(j + 0.5, i + 0.5, str(int(board_array[i][j])), 
                        ha='center', va='center', fontweight='bold', fontsize=8)
        
        ax.set_title(title, fontsize=14)    

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.geometry('+500+120')

        plt.tight_layout()

        if show:
            plt.show(block=True)

        return fig, ax
            

def main():
    game_environment = GameEnvironment(difficulty='hard')
    board = game_environment.place_mines((5,8))
    board = game_environment.compute_clue_values()
    game_environment.visualize_board(board, "True Board", show=False)
    game_environment.reveal((8,9))
    game_environment.visualize_board(game_environment.mask_board, "Mask Board After Reveal", show=False)

    plt.show()

if __name__ == '__main__':
    main()