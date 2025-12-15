import torch
from tqdm.auto import tqdm

from Minesweeper.evaluate.evaluate_task1 import evaluate_logic_bot
from Minesweeper.models.ActorCriticNet import NeuralBot

def evaluate_critic_bot(critic_model_path, num_games=100, device="cpu", difficulty="medium"):
    wins = 0
    total_safe = 0
    total_mines = 0

    for _ in tqdm(range(num_games), desc="Evaluating CriticBot"):
        bot = NeuralBot(critic_model_path, device=device, difficulty=difficulty)
        steps_survived, won = bot.solve()

        if won:
            wins += 1
            mines_triggered = 0
        else:
            mines_triggered = 1

        total_safe += steps_survived
        total_mines += mines_triggered

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

    # Evaluate Neural Net
    print("\nEvaluating Critic Net...")
    critic_model_path = f'Minesweeper/checkpoints/critic_model_{difficulty}_v0.pth'
    critic_results = evaluate_critic_bot(critic_model_path, num_games=num_games, device=device, difficulty=difficulty)

    print(f"Results for {difficulty} difficulty for {num_games} games")
    print("LogicBot Results:", logicbot_results)
    print("CriticNet Results:", critic_results)

if __name__ == "__main__":
    main()