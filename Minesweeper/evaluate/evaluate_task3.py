import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from tqdm.auto import tqdm

from Minesweeper.evaluate.evaluate_task1 import evaluate_logic_bot
from Minesweeper.models.ThinkingNet import ThinkingNet, play_one_game_thinking
from Minesweeper.train.task3_train_test import load_checkpoint
from Minesweeper.dataset.Task1Dataset import Task1Dataset
from Minesweeper.GameEnvironment import GameEnvironment
import Minesweeper.dataset.board_generation_utils as bg_utils


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


def analyze_loss_over_time(model, val_loader, device, difficulty='medium', steps=10):
    '''Analyze how the loss changes over the internal thinking steps.'''
    print("Analyzing Loss over Thinking Steps...")
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Accumulate loss for each step (0 to steps-1)
    step_losses = torch.zeros(steps).to(device)
    total_samples = 0
    
    weight_dict = {'easy': 10.0, 'medium': 5.0, 'hard': 4.0}
    class_weight = weight_dict.get(difficulty, 5.0)

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Calculating Loss"):
            inputs = inputs.to(device)
            if labels.dim() == 3: 
                labels = labels.unsqueeze(1)
            labels = labels.to(device).float()

            outputs = model(inputs, steps=steps)
            
            valid = (labels != -1.0)
            targets = torch.clamp(labels, 0.0, 1.0)
            weights = torch.ones_like(targets)
            weights[targets == 0.0] = class_weight

            # Calculate loss for each step t
            for t in range(steps):
                pred_t = outputs[:, t]
                loss_map = criterion(pred_t, targets) * weights
                loss = (loss_map * valid).sum() / valid.sum().clamp_min(1)
                step_losses[t] += loss * inputs.size(0)
            
            total_samples += inputs.size(0)

    avg_losses = (step_losses / total_samples).cpu().numpy()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, steps + 1), avg_losses, marker='o', linestyle='-', color='b')
    plt.title(f"Loss vs. Thinking Time ({difficulty})")
    plt.xlabel("Thinking Step")
    plt.ylabel("BCE Loss")
    plt.grid(True)
    plt.savefig(f"Minesweeper/plots/thinking_loss_{difficulty}.png")
    plt.show()

def play_game(model, steps, difficulty, device):
    """Play one game with a fixed number of thinking steps."""
    board = GameEnvironment(difficulty=difficulty)
    start = (random.randint(0, board.size - 1), random.randint(0, board.size - 1))
    board.place_mines(start)
    board.compute_clue_values()
    board.reveal(start)

    while not board.won_game() and not board.lost:
        encoded = bg_utils.encode_mask_board(board).unsqueeze(0).to(device)
        
        # Predict with specific steps
        with torch.no_grad():
            outputs = model(encoded, steps=steps)
            final_pred = outputs[0, -1] # Last step
            probs = torch.sigmoid(final_pred)[0]

        # Policy
        mask = board.mask_board
        hidden = (mask == board.HIDDEN)
        if not hidden.any(): break
        
        safe_probs = probs.clone()
        safe_probs[~hidden] = -float('inf')
        
        move_idx = torch.argmax(safe_probs).item()
        move = (move_idx // board.size, move_idx % board.size)
        
        board.reveal(move)

    return board.won_game()

def analyze_performance_over_time(model, device, difficulty='medium', max_steps=10, games_per_step=50):
    print("Analyzing Win Rate over Thinking Steps...")
    model.eval()
    
    # Test these step counts
    steps_to_test = [1, 3, 5, 8, max_steps]
    win_rates = []

    for s in steps_to_test:
        wins = 0
        for _ in tqdm(range(games_per_step), desc=f"Simulating (steps={s})"):
            if play_game(model, s, difficulty, device):
                wins += 1
        rate = wins / games_per_step
        win_rates.append(rate)
        print(f"Steps: {s} | Win Rate: {rate:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps_to_test, win_rates, marker='s', linestyle='-', color='g')
    plt.title(f"Win Rate vs. Thinking Time ({difficulty})")
    plt.xlabel("Thinking Steps")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.savefig(f"Minesweeper/plots/thinking_performance_{difficulty}.png")
    plt.show()

def visualize_thought_process(model, val_loader, device, difficulty='medium', steps=10):
    print("Generating Heatmap Visualization...")
    model.eval()
    
    # Get a single batch
    inputs, labels = next(iter(val_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs, steps=steps)
    
    # Pick the first sample in batch
    idx = 0
    sample_outputs = torch.sigmoid(outputs[idx]).cpu().numpy() 
    
    # Extract ground truth 
    ground_truth = labels[idx].cpu().numpy()
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[0] 
    
    board_size = ground_truth.shape[0] 
    
    # Select specific time points to visualize
    time_points = [0, steps//2 - 1, steps - 1]
    
    fig, axes = plt.subplots(1, len(time_points) + 1, figsize=(16, 4))
    
    for i, t in enumerate(time_points):
        ax = axes[i]
        heatmap = sample_outputs[t, 0]
        sns.heatmap(heatmap, ax=ax, cmap="RdYlBu", vmin=0, vmax=1, cbar=False)
        ax.set_title(f"Thinking Step {t+1}")
        ax.axis('off')

    # Plot Truth
    ax = axes[-1]
    truth_map = np.full_like(ground_truth, 0.5, dtype=float)
    truth_map[ground_truth == 1.0] = 1.0
    truth_map[ground_truth == 0.0] = 0.0
    
    sns.heatmap(truth_map, ax=ax, cmap="RdYlBu", vmin=0, vmax=1, cbar=True)
    ax.set_title("Ground Truth (Target)")
    ax.axis('off')

    plt.suptitle(f"Evolution of Predictions over {steps} Steps")
    plt.tight_layout()
    plt.savefig(f"Minesweeper/plots/thinking_heatmap_{difficulty}.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    difficulty = "medium"
    num_games = 100
    steps = 10

    # Evaluate LogicBot
    print("Evaluating LogicBot...")
    logicbot_results = evaluate_logic_bot(num_games=num_games, difficulty=difficulty)

    # Load trained model
    thinking_model_path = f'Minesweeper/checkpoints/thinking_{difficulty}_model.pth'
    thinking_model = ThinkingNet(device=device)
    load_checkpoint(thinking_model_path, thinking_model, device=device)

    # Evaluate Neural Net
    print("\nEvaluating Thinking Net...")
    thinking_results = evaluate_thinking_net(thinking_model, num_games=num_games, device=device, difficulty=difficulty, steps=steps)

    print(f"Results for {difficulty} difficulty for {num_games} games")
    print("LogicBot Results:", logicbot_results)
    print("ThinkingNet Results:", thinking_results)

    # Load data
    val_dataset = Task1Dataset(num_samples=100, difficulty=difficulty)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    analyze_loss_over_time(thinking_model, val_loader, device, difficulty, steps)
    visualize_thought_process(thinking_model, val_loader, device, difficulty, steps)
    analyze_performance_over_time(thinking_model, device, difficulty, max_steps=steps)

if __name__ == "__main__":
    main()