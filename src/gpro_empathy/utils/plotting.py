"""Training metrics plotting utilities."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def load_training_logs(output_dir: str) -> List[Dict]:
    """Load training logs from trainer_state.json or log files."""
    logs = []

    # Try to load from trainer_state.json
    trainer_state_path = Path(output_dir) / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path, "r") as f:
            state = json.load(f)
            if "log_history" in state:
                logs = state["log_history"]

    # If no logs found, try loading from tensorboard logs
    if not logs:
        log_dir = Path(output_dir)
        for log_file in log_dir.glob("**/events.out.tfevents.*"):
            print(f"Found TensorBoard log: {log_file}")
            print("Use TensorBoard to view: tensorboard --logdir", output_dir)
            break

    return logs


def extract_metrics(logs: List[Dict]) -> Dict[str, List[float]]:
    """Extract metrics from training logs."""
    metrics = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "learning_rate": [],
        "kl": [],
        "reward": [],
        "reward_std": [],
        "semantic_reward_mean": [],
        "empathy_reward_mean": [],
        "completion_length": [],
    }

    for log in logs:
        if "loss" not in log:
            continue

        step = log.get("step", len(metrics["step"]))
        metrics["step"].append(step)
        metrics["loss"].append(log.get("loss", 0))
        metrics["grad_norm"].append(log.get("grad_norm", 0))
        metrics["learning_rate"].append(log.get("learning_rate", 0))
        metrics["kl"].append(log.get("kl", 0))
        metrics["reward"].append(log.get("reward", 0))
        metrics["reward_std"].append(log.get("reward_std", 0))
        metrics["semantic_reward_mean"].append(
            log.get("rewards/semantic_sts_reward/mean", 0)
        )
        metrics["empathy_reward_mean"].append(
            log.get("rewards/empathy_model_reward/mean", 0)
        )
        metrics["completion_length"].append(log.get("completion_length", 0))

    return metrics


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot training metrics in a grid layout."""
    if not metrics["step"]:
        print("No metrics to plot!")
        return

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("GRPO Empathy Training Metrics", fontsize=14, fontweight="bold")

    steps = metrics["step"]

    # Plot 1: Rewards
    ax = axes[0, 0]
    ax.plot(steps, metrics["semantic_reward_mean"], label="Semantic", color="blue", linewidth=2)
    ax.plot(steps, metrics["empathy_reward_mean"], label="Empathy", color="green", linewidth=2)
    ax.plot(steps, metrics["reward"], label="Combined", color="red", linewidth=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Reward Std
    ax = axes[0, 1]
    ax.plot(steps, metrics["reward_std"], color="purple", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Std")
    ax.set_title("Reward Standard Deviation")
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss
    ax = axes[0, 2]
    ax.plot(steps, metrics["loss"], color="red", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Plot 4: KL Divergence
    ax = axes[1, 0]
    ax.plot(steps, metrics["kl"], color="orange", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence from Reference")
    ax.grid(True, alpha=0.3)

    # Plot 5: Gradient Norm
    ax = axes[1, 1]
    ax.plot(steps, metrics["grad_norm"], color="brown", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    # Plot 6: Learning Rate
    ax = axes[1, 2]
    ax.plot(steps, metrics["learning_rate"], color="teal", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Plot 7: Completion Length
    ax = axes[2, 0]
    ax.plot(steps, metrics["completion_length"], color="navy", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens")
    ax.set_title("Average Completion Length")
    ax.grid(True, alpha=0.3)

    # Plot 8: Semantic vs Empathy Scatter
    ax = axes[2, 1]
    scatter = ax.scatter(
        metrics["semantic_reward_mean"],
        metrics["empathy_reward_mean"],
        c=steps,
        cmap="viridis",
        alpha=0.7,
        s=50,
    )
    ax.set_xlabel("Semantic Reward")
    ax.set_ylabel("Empathy Reward")
    ax.set_title("Semantic vs Empathy Rewards")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Step")

    # Plot 9: Summary Statistics
    ax = axes[2, 2]
    ax.axis("off")

    # Calculate summary stats
    summary_text = (
        f"Training Summary\n"
        f"{'─' * 30}\n"
        f"Total Steps: {len(steps)}\n"
        f"Final Reward: {metrics['reward'][-1]:.4f}\n"
        f"Avg Semantic: {np.mean(metrics['semantic_reward_mean']):.4f}\n"
        f"Avg Empathy: {np.mean(metrics['empathy_reward_mean']):.4f}\n"
        f"Final KL: {metrics['kl'][-1]:.6f}\n"
        f"Avg Completion: {np.mean(metrics['completion_length']):.1f} tokens\n"
        f"{'─' * 30}\n"
        f"Reward Trend: {_get_trend(metrics['reward'])}\n"
        f"KL Trend: {_get_trend(metrics['kl'])}"
    )
    ax.text(
        0.1, 0.5, summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _get_trend(values: List[float]) -> str:
    """Determine if values are trending up, down, or stable."""
    if len(values) < 2:
        return "N/A"

    first_half = np.mean(values[:len(values)//2])
    second_half = np.mean(values[len(values)//2:])

    diff = second_half - first_half
    threshold = 0.01 * abs(first_half) if first_half != 0 else 0.01

    if diff > threshold:
        return "Increasing"
    elif diff < -threshold:
        return "Decreasing"
    else:
        return "Stable"


def plot_from_output_dir(
    output_dir: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Convenience function to plot metrics directly from output directory."""
    logs = load_training_logs(output_dir)

    if not logs:
        print(f"No training logs found in {output_dir}")
        print("Make sure training has completed and logs are saved.")
        return

    metrics = extract_metrics(logs)

    if save_path is None:
        save_path = os.path.join(output_dir, "training_metrics.png")

    plot_training_metrics(metrics, save_path=save_path, show=show)


def plot_reward_distribution(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot reward score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Reward Score Distributions", fontsize=12, fontweight="bold")

    # Semantic rewards
    ax = axes[0]
    ax.hist(metrics["semantic_reward_mean"], bins=20, color="blue", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(metrics["semantic_reward_mean"]), color="red", linestyle="--", label=f'Mean: {np.mean(metrics["semantic_reward_mean"]):.3f}')
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Semantic Similarity")
    ax.legend()

    # Empathy rewards
    ax = axes[1]
    ax.hist(metrics["empathy_reward_mean"], bins=20, color="green", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(metrics["empathy_reward_mean"]), color="red", linestyle="--", label=f'Mean: {np.mean(metrics["empathy_reward_mean"]):.3f}')
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Empathy Model")
    ax.legend()

    # Combined rewards
    ax = axes[2]
    ax.hist(metrics["reward"], bins=20, color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(metrics["reward"]), color="red", linestyle="--", label=f'Mean: {np.mean(metrics["reward"]):.3f}')
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Combined Reward")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Distribution plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
