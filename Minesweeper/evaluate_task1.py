import random
from tqdm.auto import tqdm
import torch

from Minesweeper.GameEnvironment import GameEnvironment
from Minesweeper.LogicBot import LogicBot
from Minesweeper.models.MinePredictionNet import MinePredictionNet, play_one_game_nn
from Minesweeper.train_test import load_checkpoint


def evaluate_logic_bot(num_games=100, difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0

    for i in tqdm(range(num_games), desc="Evaluating LogicBot"):
        bot = LogicBot(difficulty=difficulty, seed=i)
        safe_opened, mines_triggered = bot.solve()

        env = bot.game_environment
        if env.won_game():
            wins += 1

        total_safe += safe_opened
        total_mines += mines_triggered

    return {
        "win_rate": wins / num_games,
        "avg_safe_moves": total_safe / num_games,
        "avg_mines_triggered": total_mines / num_games,
    }

def evaluate_neural_net(model, num_games=100, device="cpu", difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0

    for _ in tqdm(range(num_games), desc="Evaluating Neural Net"):
        won, safe_moves, mines = play_one_game_nn(model, device=device, difficulty=difficulty)
        wins += int(won)
        total_safe += safe_moves
        total_mines += mines

    return {
        "win_rate": wins / num_games,
        "avg_safe_moves": total_safe / num_games,
        "avg_mines_triggered": total_mines / num_games,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    model = MinePredictionNet(input_size=(12, 22, 22), device=device)
    load_checkpoint('Minesweeper/checkpoints/medium_model.pth', model=model, device=device)

    # Evaluate LogicBot
    logicbot_results = evaluate_logic_bot(num_games=100, difficulty="medium")
    print("LogicBot Results:", logicbot_results)

    # Evaluate Neural Net
    nn_results = evaluate_neural_net(model, num_games=100, device=device, difficulty="medium")
    print("Neural Net Results:", nn_results)