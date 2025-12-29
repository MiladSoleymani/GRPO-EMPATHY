#!/usr/bin/env python3
"""
Main training script for GRPO Empathy model.
"""
import argparse
import yaml
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpro_empathy.training.grpo_trainer import GPROEmpathyTrainer
from gpro_empathy.utils.plotting import plot_from_output_dir, plot_reward_distribution, extract_metrics, load_training_logs


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train GPRO Empathy model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max training steps from config"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        help="Override save steps from config"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate training plots after training (default: True)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable training plots"
    )

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config values with command line arguments
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.save_steps:
        config['training']['save_steps'] = args.save_steps
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    print("=== GPRO Empathy Training ===")
    print(f"Model: {config['model']['name']}")
    print(f"Max steps: {config['training']['max_steps']}")
    print(f"Output dir: {config['training']['output_dir']}")
    print(f"LoRA rank: {config['model']['lora_rank']}")
    
    # Initialize trainer
    trainer = GPROEmpathyTrainer(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        lora_rank=config['model']['lora_rank'],
        load_in_4bit=config['model']['load_in_4bit'],
        fast_inference=config['model']['fast_inference'],
        gpu_memory_utilization=config['model']['gpu_memory_utilization'],
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=config['training']['learning_rate'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        logging_steps=config['training']['logging_steps'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_generations=config['training']['num_generations'],
        max_steps=config['training']['max_steps'],
        save_steps=config['training']['save_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        output_dir=config['training']['output_dir'],
        max_prompt_length=config['training']['max_prompt_length'],
    )
    
    print("\n=== Starting Training ===")
    # Start training
    result = trainer.train()
    
    print("\n=== Training Complete ===")
    
    # Save LoRA adapter
    lora_path = config['paths']['lora_save_path']
    print(f"Saving LoRA adapter to: {lora_path}")
    trainer.save_lora(lora_path)
    
    # Test inference
    print("\n=== Testing Inference ===")
    
    # Import the system prompt to test proper empathy reasoning
    from gpro_empathy.data.dataset_loader import get_system_prompt
    
    # Create test prompts with system prompt (like in training)
    test_message = "I'm feeling really overwhelmed with work lately."
    test_prompt = trainer.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": test_message}
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    print(f"Test Input: '{test_message}'")
    print("Expected: Model should analyze emotion and generate reasoning + empathetic response")
    
    # Generate without LoRA
    print("Generating without LoRA:")
    output1 = trainer.generate_sample(test_prompt, lora_request=None)
    print(output1[:500])
    
    # Load LoRA and generate
    print("\nGenerating with trained LoRA:")
    lora_req = trainer.load_lora(lora_path)
    output2 = trainer.generate_sample(test_prompt, lora_request=lora_req)
    print(output2[:500])
    
    # Generate training plots
    if args.plot and not args.no_plot:
        print("\n=== Generating Training Plots ===")
        try:
            output_dir = config['training']['output_dir']

            # Plot main metrics
            plot_path = os.path.join(output_dir, "training_metrics.png")
            plot_from_output_dir(output_dir, save_path=plot_path, show=False)

            # Plot reward distributions
            logs = load_training_logs(output_dir)
            if logs:
                metrics = extract_metrics(logs)
                dist_path = os.path.join(output_dir, "reward_distributions.png")
                plot_reward_distribution(metrics, save_path=dist_path, show=False)

            print(f"Training plots saved to: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    print(f"\n=== Training completed successfully! ===")
    print(f"LoRA adapter saved to: {lora_path}")
    print(f"Training outputs in: {config['training']['output_dir']}")


if __name__ == "__main__":
    main()