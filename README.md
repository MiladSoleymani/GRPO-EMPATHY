# GRPO Empathy

Reinforcement-learning fine-tuning of an LLM to give empathetic replies, using Group Relative Policy Optimization (GRPO) with LoRA. The model first reasons about the user's emotion and then answers, and it is rewarded by two learned reward models.

## Method

- **Policy:** Llama 3.1 8B Instruct (or DeepSeek-R1-Distill-Llama-8B), 4-bit, LoRA rank 32, trained with Unsloth, TRL and vLLM.
- **Rewards:**
  - *Semantic similarity* between the user message and the answer (`cross-encoder/stsb-roberta-large`), calibrated to [0, 1].
  - *Empathy score* from a RoBERTa classifier fine-tuned on WASSA empathy data (`miladsolo/roberta-lora-wassa-empathy`).
- **Data:** `miladsolo/wassa-conv-turn-empathy`, conversation turns from WASSA.
- **Output format:** `<reasoning>` (emotion, intensity, plan) followed by `<answer>`.
- RLOO (REINFORCE Leave-One-Out) is available as an alternative trainer.

## Installation

```bash
git clone https://github.com/MiladSoleymani/GRPO-EMPATHY.git
cd GRPO-EMPATHY
pip install -r requirements.txt
pip install -e .
```

A CUDA GPU is required.

## Usage

```bash
# Train (settings in configs/training_config.yaml)
python scripts/train.py --config configs/training_config.yaml --max-steps 500

# Use RLOO instead of GRPO
python scripts/train.py --trainer-type rloo

# Inference
python scripts/inference.py --message "I'm feeling really overwhelmed with work lately."
python scripts/inference.py --interactive
```

## Project structure

```
src/gpro_empathy/   data loading, reward functions, GRPO/RLOO trainers, inference
scripts/            train, inference and plotting
configs/            training configuration
notebooks/          dataset preparation and RoBERTa empathy-model fine-tuning
docs/               technical overview and empathy-model training notes
```

## License

MIT
