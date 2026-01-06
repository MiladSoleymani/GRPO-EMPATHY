import unsloth
import torch
import json
import os
from unsloth import FastLanguageModel
from trl import RLOOConfig, RLOOTrainer
from transformers import TrainerCallback
from typing import List, Callable, Dict, Any, Optional

from ..data.dataset_loader import load_wassa_empathy
from ..models.reward_functions import semantic_sts_reward, empathy_model_reward


class LoggingCallback(TrainerCallback):
    """Callback to capture and save training logs for plotting."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_entry = {"step": state.global_step, **logs}
            self.logs.append(log_entry)

    def on_train_end(self, args, state, control, **kwargs):
        self.save_logs()

    def save_logs(self):
        """Save collected logs to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "training_logs.json")
        with open(log_path, "w") as f:
            json.dump({"log_history": self.logs}, f, indent=2)
        print(f"Training logs saved to: {log_path}")


class RLOOEmpathyTrainer:
    """RLOO trainer for empathy-based language model training.

    RLOO (REINFORCE Leave-One-Out) uses a leave-one-out baseline for variance
    reduction, which can be more sample-efficient than standard REINFORCE.

    Reference: "Back to Basics: Revisiting REINFORCE Style Optimization for
    Learning from Human Feedback in LLMs" (Ahmadian et al., 2024)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length: int = 1024,
        lora_rank: int = 32,
        load_in_4bit: bool = True,
        fast_inference: bool = True,
        gpu_memory_utilization: float = 0.6,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank

        # Initialize model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=int(max_seq_length),
            load_in_4bit=bool(load_in_4bit),
            fast_inference=bool(fast_inference),
            max_lora_rank=int(lora_rank),
            gpu_memory_utilization=float(gpu_memory_utilization),
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=int(lora_rank),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=int(lora_rank),
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        self.trainer = None
        self.output_dir = None
        self.logging_callback = None

    def setup_training(
        self,
        learning_rate: float = 5e-6,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.99,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        optim: str = "paged_adamw_8bit",
        logging_steps: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        num_generations: int = 6,
        max_steps: int = 250,
        save_steps: int = 250,
        max_grad_norm: float = 0.1,
        output_dir: str = "outputs",
        reward_funcs: Optional[List[Callable]] = None,
        max_prompt_length: int = 256,
        # RLOO-specific parameters
        beta: float = 0.04,  # KL penalty coefficient
        epsilon: float = 0.2,  # Clipping parameter for importance sampling
    ):
        """Setup RLOO training configuration.

        Args:
            beta: KL penalty coefficient to prevent policy from deviating too far
                from reference policy. Default: 0.04
            epsilon: Clipping parameter for importance sampling ratio to prevent
                large policy updates. Default: 0.2
        """

        if reward_funcs is None:
            reward_funcs = [semantic_sts_reward, empathy_model_reward]

        training_args = RLOOConfig(
            learning_rate=float(learning_rate),
            adam_beta1=float(adam_beta1),
            adam_beta2=float(adam_beta2),
            weight_decay=float(weight_decay),
            warmup_ratio=float(warmup_ratio),
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            logging_steps=int(logging_steps),
            per_device_train_batch_size=int(per_device_train_batch_size),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            num_generations=int(num_generations),
            max_prompt_length=int(max_prompt_length),
            max_completion_length=self.max_seq_length - int(max_prompt_length),
            max_steps=int(max_steps),
            save_steps=int(save_steps),
            max_grad_norm=float(max_grad_norm),
            report_to="tensorboard",
            output_dir=output_dir,
            # RLOO-specific parameters
            beta=float(beta),
            epsilon=float(epsilon),
        )

        # Load dataset
        dataset = load_wassa_empathy()

        # Store output_dir and create logging callback
        self.output_dir = output_dir
        self.logging_callback = LoggingCallback(output_dir)

        self.trainer = RLOOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            callbacks=[self.logging_callback],
        )

    def train(self):
        """Start training."""
        if self.trainer is None:
            raise ValueError("Training not setup. Call setup_training() first.")

        return self.trainer.train()

    def save_lora(self, path: str):
        """Save LoRA adapter."""
        self.model.save_lora(path)

    def load_lora(self, path: str):
        """Load LoRA adapter for inference."""
        return self.model.load_lora(path)

    def generate_sample(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        lora_request=None,
    ) -> str:
        """Generate a sample using the trained model."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        output = (
            self.model.fast_generate(
                [prompt],
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0]
            .outputs[0]
            .text
        )

        return output
