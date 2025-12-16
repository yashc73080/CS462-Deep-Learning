import torch
from tqdm.auto import tqdm

from Minesweeper.evaluate.evaluate_task1 import evaluate_logic_bot
from Minesweeper.models.ThinkingNet import ThinkingNet, play_one_game_thinking
from Minesweeper.train.task3_train_test import load_checkpoint

def evaluate_thinking_net(model, num_games=100, device="cpu", difficulty="medium", steps=10):
    wins = 0
    total_safe = 0
    total_mines = 0

    for i in tqdm(range(num_games), desc="Playing game"):
        won, safe_moves, mines = play_one_game_thinking(model, device=device, difficulty=difficulty, steps=steps)
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
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    difficulty = "medium"
    num_games = 100

    # Evaluate LogicBot
    print("Evaluating LogicBot...")
    logicbot_results = evaluate_logic_bot(num_games=num_games, difficulty=difficulty)

    # Load trained model
    thinking_model_path = f'Minesweeper/checkpoints/thinking_{difficulty}_model.pth'
    thinking_model = ThinkingNet(device=device)
    load_checkpoint(thinking_model_path, thinking_model, device=device)

    # Evaluate Neural Net
    print("\nEvaluating Thinking Net...")
    thinking_results = evaluate_thinking_net(thinking_model, num_games=num_games, device=device, difficulty=difficulty, steps=10)

    print(f"Results for {difficulty} difficulty for {num_games} games")
    print("LogicBot Results:", logicbot_results)
    print("ThinkingNet Results:", thinking_results)

if __name__ == "__main__":
    main()