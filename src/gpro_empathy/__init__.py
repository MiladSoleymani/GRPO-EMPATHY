"""
GRPO Empathy: Policy Optimization for Empathy Training

Supports two training algorithms:
- GRPO: Group Relative Policy Optimization
- RLOO: REINFORCE Leave-One-Out
"""

__version__ = "0.1.0"
__author__ = "Milad Soleymani"
__email__ = "your.email@example.com"

from .training.grpo_trainer import GPROEmpathyTrainer
from .training.rloo_trainer import RLOOEmpathyTrainer
from .utils.inference import EmpathyInference
from .utils.plotting import (
    plot_training_metrics,
    plot_from_output_dir,
    plot_reward_distribution,
)
from .data.dataset_loader import load_wassa_empathy, get_system_prompt
from .models.reward_functions import (
    semantic_sts_reward,
    empathy_model_reward,
    SemanticSimilarityReward,
    EmpathyModelReward,
)

__all__ = [
    "GPROEmpathyTrainer",
    "RLOOEmpathyTrainer",
    "EmpathyInference",
    "load_wassa_empathy",
    "get_system_prompt",
    "semantic_sts_reward",
    "empathy_model_reward",
    "SemanticSimilarityReward",
    "EmpathyModelReward",
    "plot_training_metrics",
    "plot_from_output_dir",
    "plot_reward_distribution",
]