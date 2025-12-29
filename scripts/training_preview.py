#!/usr/bin/env python3
"""
Training preview script - shows one training step before full training.
"""
import argparse
import yaml
import os
import torch
from pathlib import Path
from vllm import SamplingParams

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpro_empathy.training.grpo_trainer import GPROEmpathyTrainer
from gpro_empathy.data.dataset_loader import load_wassa_empathy
from gpro_empathy.models.reward_functions import (
    semantic_sts_reward,
    empathy_model_reward,
)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def show_training_preview(config):
    """Show a preview of training inputs, outputs, and reward scores."""
    print("=== GPRO Empathy Training Preview ===")
    print(f"Model: {config['model']['name']}")
    print(f"LoRA rank: {config['model']['lora_rank']}")

    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = GPROEmpathyTrainer(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        lora_rank=config["model"]["lora_rank"],
        load_in_4bit=config["model"]["load_in_4bit"],
        fast_inference=config["model"]["fast_inference"],
        gpu_memory_utilization=config["model"]["gpu_memory_utilization"],
    )

    # Load dataset
    print("📚 Loading dataset...")
    dataset = load_wassa_empathy()

    # Get a few sample prompts
    sample_size = 3
    samples = dataset.select(range(sample_size))

    print(f"\n=== Training Step Preview ({sample_size} samples) ===")

    # Extract prompts for generation
    prompts = []

    for i, sample in enumerate(samples):
        prompt = trainer.tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

        print(f"\n--- Sample {i+1} ---")
        print("🎯 Input (User Message):")
        user_content = sample["prompt"][-1]["content"]
        print(f"   {user_content}")
        print("🤖 Model Task: Generate reasoning + empathetic response")

    # Generate responses
    print("\n🤖 Generating model responses...")
    sampling_params = SamplingParams(
        temperature=config.get("inference", {}).get("temperature", 0.8),
        top_p=config.get("inference", {}).get("top_p", 0.95),
        max_tokens=1024,  # Enough for reasoning + answer sections
    )

    try:
        outputs = trainer.model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=None,
        )

        completions = [output.outputs[0].text for output in outputs]

        # Show generated responses
        for i, completion in enumerate(completions):
            print(f"\n--- Sample {i+1} Response ---")
            print("🤖 Generated Response:")
            print(f"   {completion}{'...' if len(completion) > 200 else ''}")

        # Calculate reward scores
        print("\n🏆 Calculating Reward Scores...")

        # Prepare prompts for reward functions (need proper format)
        reward_prompts = [sample["prompt"] for sample in samples]

        try:
            # Calculate semantic similarity rewards
            print("   📐 Computing semantic similarity scores...")
            semantic_scores = semantic_sts_reward(reward_prompts, completions)

            # Calculate empathy rewards
            print("   💝 Computing empathy scores...")
            empathy_scores = empathy_model_reward(reward_prompts, completions)

            # Display scores
            print("\n=== REWARD SCORES ===")
            for i in range(len(samples)):
                sample = samples[i]
                user_content = sample["prompt"][-1]["content"]

                print(f"\n--- Sample {i+1} Scores ---")
                print(f"🎯 Input: {user_content}")
                print(
                    f"🤖 Response: {completions[i]}{'...' if len(completions[i]) > 100 else ''}"
                )
                print(f"📐 Semantic Similarity: {semantic_scores[i]:.4f}")
                print(f"💝 Empathy Score: {empathy_scores[i]:.4f}")
                print(
                    f"🔄 Combined Score: {(semantic_scores[i] + empathy_scores[i]) / 2:.4f}"
                )

        except Exception as e:
            print(f"⚠️  Error calculating rewards: {e}")
            print(
                "   This is normal for the first run - reward models need to initialize"
            )

    except Exception as e:
        print(f"⚠️  Error generating responses: {e}")
        print("   This might indicate GPU memory issues or model loading problems")

    print(f"\n=== Preview Complete ===")
    print("This shows how GRPO training works:")
    print("1. 📝 Input prompts are formatted with system instructions")
    print("2. 🤖 Model generates multiple responses per prompt")
    print("3. 🏆 Reward functions score each response")
    print("4. 📈 GRPO uses these scores to improve the model")
    print("5. 🔄 Process repeats for multiple steps")


def main():
    parser = argparse.ArgumentParser(description="Preview GPRO Empathy training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Only show preview, don't start training",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Show training preview
    show_training_preview(config)

    if not args.no_training:
        print("\n" + "=" * 50)
        print("🚀 Starting Full Training Process...")
        print("=" * 50)

        # Import and run the main training script
        from train import main as train_main

        # Save original argv and set new one for training
        original_argv = sys.argv
        sys.argv = ["train.py", "--config", args.config]

        try:
            train_main()
        finally:
            sys.argv = original_argv
    else:
        print("\n✅ Preview completed. Use --no-training=false to start training.")


if __name__ == "__main__":
    main()
