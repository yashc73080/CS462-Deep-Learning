import random

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

        self.safe_open_count = 0
        self.game_over = False

    # ---- Helpers ---
    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def update_after_reveal(self):
        mask = self.game_environment.mask_board

        for x in range(self.size):
            for y in range(self.size):
                value = int(mask[x][y])

                if value != self.HIDDEN:

                    # Remove from remaining list
                    if (x, y) in self.cells_remaining:
                        self.cells_remaining.remove((x, y))

                    # Store clue numbers (including zero)
                    if value >= 0:
                        self.clue_number[(x, y)] = value

    def run_inference(self):
        '''
        Runs inference rules until no new information can be inferred
        '''
        mask = self.game_environment.mask_board

        changed = True
        while changed:
            changed = False

            for (x, y), clue in list(self.clue_number.items()):
                neighbors = self.get_neighbors((x, y))

                hidden = [n for n in neighbors if n in self.cells_remaining and n not in self.inferred_mine]
                mines = [n for n in neighbors if n in self.inferred_mine]
                safe = [n for n in neighbors if n in self.inferred_safe or mask[n[0], n[1]] >= 0]
                # revealed safe includes: clue numbers and zeros

                # Rule A — Identify Mines
                if clue - len(mines) == len(hidden) and len(hidden) > 0:
                    for n in hidden:
                        if n not in self.inferred_mine:
                            self.inferred_mine.add(n)
                            if n in self.cells_remaining:
                                self.cells_remaining.remove(n)
                            changed = True

                # Rule B — Identify Safes
                total_neighbors = len(neighbors)
                if (total_neighbors - clue) - len(safe) == len(hidden) and len(hidden) > 0:
                    for n in hidden:
                        if n not in self.inferred_safe:
                            self.inferred_safe.add(n)
                            if n in self.cells_remaining:
                                self.cells_remaining.remove(n)
                            changed = True


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
    for i in range(n):
        bot = LogicBot(difficulty=difficulty, seed=i)
        safe_opened, mines_triggered = bot.solve()

        env = bot.game_environment
        if env.won_game():
            wins += 1

        total_safe += safe_opened
        total_mines += mines_triggered

        # mask_board should only contain HIDDEN (-2), mines (-1 only if game over reveal), or >=0
        assert ((env.mask_board == env.HIDDEN) |
                (env.mask_board == env.MINE) |
                (env.mask_board >= 0)).all().item()

    print(f"difficulty={difficulty} trials={n} wins={wins} win_rate={wins/n:.3f}")


def main():
    for difficulty in ['easy', 'medium', 'hard']:
        run_trials(n=15, difficulty=difficulty)
    
if __name__ == "__main__":
    main()

# Run with: 
# $ python -m Minesweeper.LogicBot