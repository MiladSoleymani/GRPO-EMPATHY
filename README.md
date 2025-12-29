# GRPO Empathy: Group Relative Policy Optimization for Empathy Training

A structured project for training empathetic language models using Group Relative Policy Optimization (GRPO) with specialized reward functions for empathy and semantic similarity.

## Overview

This project implements GRPO training for developing more empathetic conversational AI models. It uses:

- **Semantic Similarity Reward**: Measures how well the model response aligns with user input
- **Empathy Model Reward**: Uses a fine-tuned RoBERTa model to evaluate empathy levels

## Project Structure

```
GPRO-EMPATHY/
├── src/gpro_empathy/           # Main package
│   ├── data/                   # Dataset loading utilities
│   ├── models/                 # Reward functions
│   ├── training/               # GRPO trainer
│   └── utils/                  # Inference utilities
├── scripts/                    # Training, inference, and preview scripts
├── configs/                    # Configuration files
├── notebooks/                  # Original development notebooks
└── requirements.txt            # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MiladSoleymani/GRPO-EMPATHY.git
cd GRPO-EMPATHY
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training

Train the model with default configuration:

```bash
python scripts/train.py
```

Or with custom config:

```bash
python scripts/train.py --config configs/training_config.yaml --max-steps 500
```

### Inference

Single message response:

```bash
python scripts/inference.py --message "I'm feeling really overwhelmed with work lately."
```

Interactive chat mode:

```bash
python scripts/inference.py --interactive
```

## Configuration

Edit `configs/training_config.yaml` to customize:

- Model parameters (LoRA rank, sequence length)
- Training hyperparameters (learning rate, batch size)
- Reward function settings
- Inference parameters

## Reward Functions

### Semantic Similarity Reward
- Uses `cross-encoder/stsb-roberta-large`
- Measures alignment between user input and model response
- Calibrated to [0,1] range

### Empathy Model Reward  
- Uses `miladsolo/roberta-lora-wassa-empathy`
- Predicts empathy level in model responses
- Based on WASSA empathy classification

## Model Architecture

- **Base Model**: meta-llama/meta-Llama-3.1-8B-Instruct
- **Fine-tuning**: LoRA with rank 32
- **Training**: GRPO with multiple reward functions
- **Inference**: vLLM for fast generation

## Output Format

The model generates responses in this format:

```
<reasoning>
- User is expressing work stress and overwhelm
- Emotion: anxiety/stress, intensity: 3-4  
- Plan: validate feelings and offer gentle support
</reasoning>
<answer>
I hear how overwhelming work has been feeling for you lately - that kind of stress can be really draining. Have you been able to take any small breaks for yourself during the day?
</answer>
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{grpo-empathy,
  title={GRPO Empathy: Group Relative Policy Optimization for Empathy Training},
  author={Milad Soleymani},
  year={2024},
  url={https://github.com/MiladSoleymani/GRPO-EMPATHY}
}
```