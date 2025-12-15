import random
import numpy as np

from Minesweeper.GameEnvironment import GameEnvironment

class LogicBot():
    def __init__(self, difficulty=None, seed=None):
        if seed is not None:
            random.seed(seed)

        if difficulty == None:
            difficulty = random.choice(['easy', 'medium', 'hard'])
        self.game_environment = GameEnvironment(difficulty=difficulty)

        self.size = self.game_environment.size
        self.HIDDEN = self.game_environment.HIDDEN
        self.MINE = self.game_environment.MINE

        # Required data structures
        self.cells_remaining = set([(x,y) for x in range(self.size) for y in range(self.size)])
        self.inferred_safe = set()
        self.inferred_mine = set()
        self.clue_number = dict() # for each opened cell, what was clue number

        self.neighbors_map = self._precompute_neighbors()

        self.safe_open_count = 0
        self.game_over = False

    # ---- Helpers ---
    def _precompute_neighbors(self):
        neighbors_map = {}
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        for x in range(self.size):
            for y in range(self.size):
                nb = []
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        nb.append((nx, ny))
                neighbors_map[(x, y)] = nb
        return neighbors_map

    def get_neighbors(self, cell):
        return self.neighbors_map[cell]
    
    def update_after_reveal(self):
        '''Updates internal state after a reveal action'''
        mask = self.game_environment.mask_board.cpu().numpy()

        # Find indices where the mask is revealed (not HIDDEN)
        rows, cols = np.where(mask != self.HIDDEN)
        
        for x, y in zip(rows, cols):
            
            # Remove from set if present
            if (x, y) in self.cells_remaining:
                self.cells_remaining.remove((x, y))
                
            # Read the clue number
            value = int(mask[x][y])
            if value >= 0:
                self.clue_number[(x, y)] = value

    def run_inference(self):
        '''
        Runs inference rules until no new information can be inferred.
        '''
        changed = True
        while changed:
            changed = False
            
            # List to track clues that are fully solved (no hidden neighbors left)
            solved_clues = []

            for (x, y), clue in list(self.clue_number.items()):
                neighbors = self.get_neighbors((x, y))

                # Don't need to check 'safe' or 'mask' explicitly.
                # All neighbors are either: 
                # 1. Unknown (in cells_remaining)
                # 2. Known Mines (in inferred_mine)
                # 3. Known Safe (everything else)
                
                hidden = []
                mines_count = 0
                
                for n in neighbors:
                    if n in self.cells_remaining:
                        hidden.append(n)
                    elif n in self.inferred_mine:
                        mines_count += 1
                
                # OPTIMIZATION: If no hidden neighbors, this clue provides no new info.
                # Mark it for removal so we don't check it again next loop.
                if not hidden:
                    solved_clues.append((x, y))
                    continue

                # Rule A - Identify Mines
                # If (clue - known_mines) equals the number of hidden cells, 
                if clue - mines_count == len(hidden):
                    for n in hidden:
                        self.inferred_mine.add(n)
                        if n in self.cells_remaining:
                            self.cells_remaining.remove(n)
                        changed = True

                # Rule B - Identify Safes
                # If (clue) equals (known_mines), all remaining hidden cells are Safe.
                elif clue == mines_count:
                    for n in hidden:
                        self.inferred_safe.add(n)
                        if n in self.cells_remaining:
                            self.cells_remaining.remove(n)
                        changed = True
            
            # Garbage Collect: Remove solved clues from the dictionary
            for k in solved_clues:
                del self.clue_number[k]


    # --- Main solving function ---
    def solve(self):
        # First click
        start = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.game_environment.place_mines(start)
        self.game_environment.compute_clue_values()
        self.game_environment.reveal(start)

        # Sync state
        self.update_after_reveal()
        self.run_inference()

        while not self.game_over:

            # Priority 1: Click safe inferred cells
            if len(self.inferred_safe) > 0:
                cell = self.inferred_safe.pop()
            else:
                # Avoid clicking known mines
                possible = self.cells_remaining - self.inferred_mine
                if len(possible) == 0:
                    print("No cells left to click.")
                    break
                cell = random.choice(list(possible))

            # Reveal
            result = self.game_environment.reveal(cell)
            if not result:
                self.game_over = True
                break
            self.safe_open_count += 1

            # Update and infer
            self.update_after_reveal()
            self.run_inference()

            if self.game_environment.won_game():
                self.game_over = True
                break

        # Compute summary statistics after game over
        # safe_open_count: exact number of non-mine cells opened, not total number of safe cells 
        mines_triggered = 1 if self.game_environment.lost else 0
        print(f"Game over. Safe opened: {self.safe_open_count}, Mines triggered: {mines_triggered}")

        return self.safe_open_count, mines_triggered
    

def run_trials(n=50, difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0

    # Accumulators for averages
    win_safe_sum = 0
    win_count = 0
    loss_safe_sum = 0
    loss_count = 0

    for i in range(n):
        bot = LogicBot(difficulty=difficulty, seed=i)
        safe_opened, mines_triggered = bot.solve()

        env = bot.game_environment
        if env.won_game():
            wins += 1
            win_count += 1
            win_safe_sum += safe_opened
        else:
            loss_count += 1
            loss_safe_sum += safe_opened

        total_safe += safe_opened
        total_mines += mines_triggered

        # mask_board should only contain HIDDEN (-2), mines (-1 only if game over reveal), or >=0
        assert ((env.mask_board == env.HIDDEN) |
                (env.mask_board == env.MINE) |
                (env.mask_board >= 0)).all().item()

    avg_safe_overall = total_safe / n if n > 0 else 0.0
    avg_safe_wins = (win_safe_sum / win_count) if win_count > 0 else 0.0
    avg_safe_losses = (loss_safe_sum / loss_count) if loss_count > 0 else 0.0

    print(f"difficulty={difficulty} trials={n} wins={wins} win_rate={wins/n:.3f}")
    print(f"avg_safe_overall={avg_safe_overall:.2f} avg_safe_wins={avg_safe_wins:.2f} avg_safe_losses={avg_safe_losses:.2f}")


def main():
    for difficulty in ['easy', 'medium', 'hard']:
        run_trials(n=100, difficulty=difficulty)
    
if __name__ == "__main__":
    main()

# Run with: 
# $ python -m Minesweeper.LogicBot