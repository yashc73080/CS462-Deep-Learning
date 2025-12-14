import torch
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import deque
import click

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
        
        self.MINE = -1
        self.HIDDEN = -2 # any hidden cell
        self.mask_board = self.initialize_board(self.HIDDEN) # The board shown to the player
        self.lost = False

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
        self.lost = False
        mines_placed = 0

        while mines_placed < self.num_mines:
            # Randomly select a cell
            mine_x = random.randint(0, self.size - 1)
            mine_y = random.randint(0, self.size - 1)

            # Place mine if it's not the first clicked cell and not already a mine
            if (mine_x, mine_y) != (x, y) and self.board[mine_x][mine_y] != self.MINE:
                self.board[mine_x][mine_y] = self.MINE
                mines_placed += 1

        return self.board
    
    def compute_clue_values(self):
        '''
        Computes the clue values for each cell in the board based on adjacent mines.
        '''
        # 1 if mine, 0 otherwise
        mines = (self.board == self.MINE).float().unsqueeze(0).unsqueeze(0)
        
        # 3x3 kernel of 1s (excluding center)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1] = 0

        # Convolve to count neighbors
        neighbor_mines = conv2d(mines, kernel, padding=1).squeeze().int()

        # Update board: Where it's NOT a mine, set the value to the neighbor count
        mask_not_mine = (self.board != self.MINE)
        self.board[mask_not_mine] = neighbor_mines[mask_not_mine]

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

        if value == self.MINE:
            # Game over: reveal full board
            self.mask_board = self.board.clone()
            self.lost = True
            # print("Clicked on a mine! Game over.")
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
            if int(self.board[x][y]) == self.MINE:
                continue  # never reveal mines with flood fill

            # Reveal current cell
            self.mask_board[x][y] = self.board[x][y]

            # If 0, expand to neighbors
            if int(self.board[x][y]) == 0:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if self.mask_board[nx][ny] == self.HIDDEN and int(self.board[nx][ny]) != self.MINE:
                            q.append((nx, ny))

    def won_game(self):
        '''
        Checks if all non-mine cells have been revealed.
        '''
        # If we ever clicked a mine, this game is a loss.
        if getattr(self, 'lost', False):
            return False

        hidden = (self.mask_board == self.HIDDEN)

        # If there are no hidden cells left, it's only a win if there were no mines
        if not hidden.any().item():
            return self.num_mines == 0

        # All remaining hidden cells must be mines.
        return bool((self.board[hidden] == self.MINE).all().item())

    
    def visualize_board(self, board=None, title="Game Board", show=True, ax=None):
        '''
        Mines are shown in red, clue values in blue gradient, empty cells in white.
        '''
        if board is None:
            board = self.board
        
        board_array = board.numpy().astype(float)
        
        # Replace mines with a high value for color mapping
        board_display = board_array.copy()
        board_display[board_display == self.MINE] = self.num_mines + 1

        # Create mask for hidden cells
        is_hidden = (board_array == self.HIDDEN)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Create heatmap
        # Use mask to hide HIDDEN cells
        sns.heatmap(board_display, cmap='YlOrRd', cbar=False, 
                    linewidths=0.5, linecolor='black', ax=ax,
                    vmin=0, vmax=self.num_mines + 1, mask=is_hidden)
        
        # Set background for hidden cells
        ax.set_facecolor('lightgray')
        
        # Add text annotations for clue values and mines
        for i in range(self.size):
            for j in range(self.size):
                if is_hidden[i][j]:
                    continue

                if board_array[i][j] == self.MINE:
                    ax.text(j + 0.5, i + 0.5, 'M', 
                        ha='center', va='center', fontsize=10)
                elif board_array[i][j] > 0:
                    ax.text(j + 0.5, i + 0.5, str(int(board_array[i][j])), 
                        ha='center', va='center', fontweight='bold', fontsize=8)
        
        ax.set_title(title, fontsize=14)    

        if show:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.geometry('+500+120')

            plt.tight_layout()
            plt.show(block=True)

        return fig, ax
    
    def play(self):
        '''
        Interactive play mode using matplotlib events for testing.
        '''
        fig, ax = plt.subplots(figsize=(8, 8))
        self.first_click_done = False

        def on_click(event):
            if event.inaxes != ax: return
            if getattr(self, 'lost', False) or self.won_game(): return

            if event.xdata is None or event.ydata is None: return
            
            # Map coordinates (xdata is col/y, ydata is row/x)
            y = int(event.xdata)
            x = int(event.ydata)
            
            if not (0 <= x < self.size and 0 <= y < self.size): return

            if not self.first_click_done:
                self.place_mines((x, y))
                self.compute_clue_values()
                self.first_click_done = True
            
            self.reveal((x, y))
            update_view()

        def update_view():
            ax.clear()
            board_to_show = self.board if getattr(self, 'lost', False) else self.mask_board
            
            title = "Minesweeper"
            if getattr(self, 'lost', False): 
                title = "GAME OVER - LOST"
            elif self.won_game(): 
                title = "CONGRATULATIONS - WON"
            
            self.visualize_board(board=board_to_show, title=title, show=False, ax=ax)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        update_view()
        plt.show()
            

@click.command()
@click.option('--difficulty', type=click.Choice(['easy', 'medium', 'hard']), default='medium', help='Difficulty level of the game.')
@click.option('--play', is_flag=True, help='Play the game interactively.')
def main(difficulty, play):
    game_environment = GameEnvironment(difficulty=difficulty)
    if play:
        game_environment.play()
    else:
        start = (random.randint(0, game_environment.size - 1), random.randint(0, game_environment.size - 1))
        print(f"First click at: {start}")
        board = game_environment.place_mines(start)
        board = game_environment.compute_clue_values()

        next = (random.randint(0, game_environment.size - 1), random.randint(0, game_environment.size - 1))
        print(f"Revealing cell at: {next}")
        game_environment.reveal(next)

        game_environment.visualize_board(board, "True Board", show=False)
        game_environment.visualize_board(game_environment.mask_board, "Mask Board After Reveal", show=False)

        plt.show()

if __name__ == '__main__':
    main()

# Run with 
# $ python Minesweeper/GameEnvironment.py