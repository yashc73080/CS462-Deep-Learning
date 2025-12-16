import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm
import random

from Minesweeper.GameEnvironment import GameEnvironment
from Minesweeper.LogicBot import LogicBot
from Minesweeper.models.MinePredictionNet import MinePredictionNet, play_one_game_nn, neural_policy
from Minesweeper.train.task1_train_test import load_checkpoint

def evaluate_logic_bot(num_games=100, difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0
    safe_moves_per_game = []

    for i in tqdm(range(num_games)):
        bot = LogicBot(difficulty=difficulty, seed=i)
        safe_opened, mines_triggered = bot.solve()

        env = bot.game_environment
        if env.won_game():
            wins += 1

        total_safe += safe_opened
        total_mines += mines_triggered
        safe_moves_per_game.append(safe_opened)

    return {
        "win_rate": wins / num_games,
        "avg_safe_moves": total_safe / num_games,
        "avg_mines_triggered": total_mines / num_games,
        "safe_moves_std": np.std(safe_moves_per_game),
    }

def evaluate_neural_net(model, num_games=100, device="cpu", difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0
    safe_moves_per_game = []

    for _ in tqdm(range(num_games)):
        won, safe_moves, mines = play_one_game_nn(model, device=device, difficulty=difficulty)
        wins += int(won)
        total_safe += safe_moves
        total_mines += mines
        safe_moves_per_game.append(safe_moves)

    return {
        "win_rate": wins / num_games,
        "avg_safe_moves": total_safe / num_games,
        "avg_mines_triggered": total_mines / num_games,
        "safe_moves_std": np.std(safe_moves_per_game),
    }


def safe_reveal(env, x, y):
    """
    Reveals a cell WITHOUT triggering the 'Game Over - Show All' behavior of the default environment.
    """
    if not (0 <= x < env.size and 0 <= y < env.size): return False, True
    if env.mask_board[x, y] != env.HIDDEN: return False, True

    value = env.board[x, y]

    # If Mine: Reveal it as a mine, but DON'T end the game
    if value == env.MINE:
        env.mask_board[x, y] = env.MINE # Manually mark it
        return True, False # It was a mine, not safe

    # If Safe: Use standard reveal
    env.reveal((x, y))
    return False, True

def run_god_mode_game(agent_type, model=None, difficulty='medium', seed=None):
    """
    Runs a game until the entire board is revealed (either safe or explicitly triggered mines).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = GameEnvironment(difficulty=difficulty)
    start = (random.randint(0, env.size - 1), random.randint(0, env.size - 1))
    env.place_mines(start)
    env.compute_clue_values()
    env.reveal(start)

    # Setup LogicBot internals if needed
    logic_bot = None
    if agent_type == 'logic':
        logic_bot = LogicBot(difficulty=difficulty) 
        logic_bot.game_environment = env
        logic_bot.size = env.size
        # Sync state
        logic_bot.cells_remaining = set([(r,c) for r in range(env.size) for c in range(env.size)])
        logic_bot.inferred_safe = set()
        logic_bot.inferred_mine = set()
        logic_bot.clue_number = dict()
        logic_bot.update_after_reveal() # Sync start

    mines_triggered = 0
    steps_survived = 0
    first_death_step = None
    
    while True:
        hidden = (env.mask_board == env.HIDDEN)
        mines = (env.board == env.MINE)
        
        if not (hidden & ~mines).any():
            break
            
        move = None
        
        if agent_type == 'logic':
            logic_bot.run_inference()
            if logic_bot.inferred_safe:
                move = logic_bot.inferred_safe.pop()
            else:
                possible = logic_bot.cells_remaining - logic_bot.inferred_mine
                if not possible: 
                    break
                move = random.choice(list(possible))
        
        elif agent_type == 'neural':
            move = neural_policy(model, env, device=next(model.parameters()).device)
            if move is None:
                candidates = [(r, c) for r in range(env.size) for c in range(env.size) if env.mask_board[r,c] == env.HIDDEN]
                if not candidates: 
                    break
                move = random.choice(candidates)

        # Execute Move (God Mode)
        is_mine, is_safe = safe_reveal(env, move[0], move[1])

        if is_mine:
            mines_triggered += 1
            if first_death_step is None:
                first_death_step = steps_survived
        else:
            if first_death_step is None:
                steps_survived += 1
            
            if agent_type == 'logic':
                logic_bot.update_after_reveal()

    return {
        "cleared": (mines_triggered == 0),
        "steps_survived": steps_survived, # Moves made before first mine
        "total_mines": mines_triggered
    }

def find_disagreement(model, device, difficulty='medium', max_attempts=100):
    print("Searching for a disagreement between LogicBot and NeuralNet...")
    
    for i in range(max_attempts):
        env = GameEnvironment(difficulty=difficulty)
        start = (random.randint(0, env.size - 1), random.randint(0, env.size - 1))
        env.place_mines(start)
        env.compute_clue_values()
        env.reveal(start)
        
        lb = LogicBot(difficulty=difficulty)
        lb.game_environment = env
        lb.update_after_reveal()
        
        # Play until they disagree
        while not env.won_game() and not env.lost:
            # 1. Get Logic Move
            lb.run_inference()
            if lb.inferred_safe:
                logic_move = list(lb.inferred_safe)[0] 
                logic_type = "Safe Inference"
            else:
                possible = lb.cells_remaining - lb.inferred_mine
                if not possible: break
                logic_move = list(possible)[0] 
                logic_type = "Random Guess"

            # 2. Get Neural Move
            nn_move = neural_policy(model, env, device=device)
            
            # 3. Compare
            if nn_move and nn_move != logic_move:
                print(f"\n--- Divergence Found (Game {i}) ---")
                print(f"LogicBot ({logic_type}): {logic_move}")
                print(f"NeuralNet: {nn_move}")
                
                # Analyze outcomes
                logic_is_mine = (env.board[logic_move] == env.MINE)
                nn_is_mine = (env.board[nn_move] == env.MINE)
                
                print(f"Logic Move Result: {'MINE' if logic_is_mine else 'SAFE'}")
                print(f"Neural Move Result: {'MINE' if nn_is_mine else 'SAFE'}")
                
                if logic_is_mine and not nn_is_mine:
                    print(">> Neural Net made a BETTER decision (Avoided Mine)")
                elif not logic_is_mine and nn_is_mine:
                    print(">> Logic Bot made a BETTER decision (Neural Net hit Mine)")
                else:
                    print(">> Both safe or both mines (Different choices).")

                # Visualize
                title = f"Divergence: Logic{logic_move} vs Neural{nn_move}"
                env.visualize_board(env.mask_board, title=title, show=True)
                return
            
            env.reveal(logic_move)
            lb.update_after_reveal()
            
    print("No significant divergences found in random search.")


def compute_stats(data_list):
    arr = np.array(data_list)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    ci = 1.96 * sem # 95% CI
    return mean, ci


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    difficulty = "medium"
    num_games = 100

    model_map = {
        'easy': 'easy_model2.pth',
        'medium': 'medium_model3.pth',
        'hard': 'medium_model3.pth',
    }

    # Load trained model
    model = MinePredictionNet(input_size=(12, 22, 22), device=device)
    load_checkpoint(f'Minesweeper/checkpoints/{model_map[difficulty]}', model=model, device=device)
    model.eval()

    # Evaluate LogicBot
    print("Evaluating LogicBot...")
    logicbot_results = evaluate_logic_bot(num_games=num_games, difficulty=difficulty)

    # Evaluate Neural Net
    print("\nEvaluating Neural Net...")
    nn_results = evaluate_neural_net(model, num_games=num_games, device=device, difficulty=difficulty)

    print(f"Results for {difficulty} difficulty for {num_games} games")
    print("LogicBot Results:", logicbot_results)
    print("Neural Net Results:", nn_results)

    # Storage
    results = {
        'logic': {'cleared': [], 'steps': [], 'mines': []},
        'neural': {'cleared': [], 'steps': [], 'mines': []}
    }

    for i in tqdm(range(num_games)):
        seed = i + 1000
        
        # Logic Run
        l_res = run_god_mode_game('logic', difficulty=difficulty, seed=seed)
        results['logic']['cleared'].append(l_res['cleared'])
        results['logic']['steps'].append(l_res['steps_survived'])
        results['logic']['mines'].append(l_res['total_mines'])
        
        # Neural Run
        n_res = run_god_mode_game('neural', model=model, difficulty=difficulty, seed=seed)
        results['neural']['cleared'].append(n_res['cleared'])
        results['neural']['steps'].append(n_res['steps_survived'])
        results['neural']['mines'].append(n_res['total_mines'])

    # Report
    print("\n" + "="*40)
    print(f"STATISTICAL REPORT (N={num_games}, Difficulty={difficulty})")
    print("="*40)
    
    metrics = [
        ("Win Rate (%)", 'cleared', 100.0),
        ("Avg Steps Survived", 'steps', 1.0),
        ("Avg Mines Triggered (God Mode)", 'mines', 1.0)
    ]
    
    for name, key, scale in metrics:
        l_mean, l_ci = compute_stats(results['logic'][key])
        n_mean, n_ci = compute_stats(results['neural'][key])
        
        print(f"\n{name}:")
        print(f"  LogicBot: {l_mean*scale:.2f} ± {l_ci*scale:.2f}")
        print(f"  NeuralNet: {n_mean*scale:.2f} ± {n_ci*scale:.2f}")
        
        diff = (n_mean - l_mean) * scale
        print(f"  Difference (Neural - Logic): {diff:+.2f}")

    # Qualitative
    print("\n" + "="*40)
    print("QUALITATIVE ANALYSIS")
    print("="*40)
    find_disagreement(model, device, difficulty)

if __name__ == "__main__":
    main()


# Run with:
# python -m Minesweeper.evaluate.evaluate_task1