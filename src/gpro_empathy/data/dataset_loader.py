from typing import Optional
from datasets import load_dataset, Dataset


SYSTEM_PROMPT = """You are a friendly, trauma-informed assistant. Analyze the user's message for emotional content and respond empathetically.

IMPORTANT: You MUST use EXACTLY this XML format with both <reasoning> and <answer> tags:

<reasoning>
- User's main concern: [what they're expressing]
- Emotional content: [emotions present]
- Intensity level: [0-5 scale]
- Response approach: [your strategy]
</reasoning>
<answer>
[Your 1-2 sentence empathetic response here]
</answer>

Example response:
<reasoning>
- User's main concern: Feeling stressed about exams
- Emotional content: Anxiety, worry
- Intensity level: 3
- Response approach: Validate feelings, offer support
</reasoning>
<answer>
It sounds like you're carrying a lot of pressure right now with your exams. That kind of stress is really tough to manage.
</answer>

Remember: Always include BOTH <reasoning> AND <answer> tags in your response.""".strip()


def _mk_instruction(utterance: str) -> str:
    """Format user text - the chat template will handle special tokens."""
    return utterance


def load_wassa_empathy(split: Optional[str] = None) -> Dataset:
    """
    Load and format WASSA empathy dataset for GRPO training.

    From your RoBERTa notebook, we know the dataset has:
    - "text" field: contains the user utterances
    - Other fields we don't need for GRPO (Emotion, EmotionalPolarity, Empathy)

    For GRPO: We only need prompts, model generates responses, reward functions score them.
    """
    # Load dataset - use train split by default
    if split is None:
        ds = load_dataset("miladsolo/wassa-conv-turn-empathy", split="train")
    else:
        ds = load_dataset("miladsolo/wassa-conv-turn-empathy", split=split)

    def _map(ex):
        # Get text from "text" field (confirmed from RoBERTa notebook)
        text = (ex.get("text") or "").strip()
        if not text:
            return {"prompt": None}

        # Format user message for Llama chat template
        user_msg = _mk_instruction(text)

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }

    # Apply mapping and filter out invalid entries
    ds = ds.map(_map)
    ds = ds.filter(lambda ex: ex["prompt"] is not None)

    # Remove unused columns to save memory
    cols_to_remove = [c for c in ds.column_names if c != "prompt"]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    print(f"✅ Prepared {len(ds)} WASSA empathy examples for GRPO training.")
    print(
        "📝 Dataset structure: Each example contains prompt for model to generate empathetic response"
    )
    print("🏆 Reward functions will score generated responses for empathy quality")

    # Quick preview
    for i in range(min(3, len(ds))):
        ex = ds[i]
        print(f"\n--- Sample {i} ---")
        user_content = ex["prompt"][-1]["content"]
        print(f"📱 User Input: '{user_content}'")
        print(
            "🤖 Model Task: Analyze emotion → Generate reasoning + empathetic response"
        )

    return ds


def get_system_prompt() -> str:
    """Get the system prompt used for training."""
    return SYSTEM_PROMPT
