import torch

from Minesweeper.LogicBot import LogicBot
from Minesweeper.models.MinePredictionNet import MinePredictionNet, play_one_game_nn
from Minesweeper.train.task1_train_test import load_checkpoint

def evaluate_logic_bot(num_games=100, difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0

    for i in range(num_games):
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

    for _ in range(num_games):
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

    difficulty = "medium"
    num_games = 100

    # Load trained model
    model = MinePredictionNet(input_size=(12, 22, 22), device=device)
    load_checkpoint(f'Minesweeper/checkpoints/{difficulty}_model3.pth', model=model, device=device)

    # Evaluate LogicBot
    print("Evaluating LogicBot...")
    logicbot_results = evaluate_logic_bot(num_games=num_games, difficulty=difficulty)

    # Evaluate Neural Net
    print("\nEvaluating Neural Net...")
    nn_results = evaluate_neural_net(model, num_games=num_games, device=device, difficulty=difficulty)

    print(f"Results for {difficulty} difficulty for {num_games} games")
    print("LogicBot Results:", logicbot_results)
    print("Neural Net Results:", nn_results)

if __name__ == "__main__":
    main()


# Run with:
# python -m Minesweeper.evaluate.evaluate_task1