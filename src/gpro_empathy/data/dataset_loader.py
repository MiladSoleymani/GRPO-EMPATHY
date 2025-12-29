from typing import Optional
from datasets import load_dataset, Dataset


SYSTEM_PROMPT = """You are a friendly, trauma-informed assistant. Analyze the user's message carefully for emotional content, intensity, and empathy needs.

When responding:
1. First, analyze what the user is expressing (concern, emotion, intensity level 0-5)
2. Then provide an empathetic response that matches their emotional needs
3. Your response should reflect their experience, and feelings
4. Keep responses to 1-2 sentences, avoid lists, quotes, or clinical tone

Output EXACTLY this format:

<reasoning>
- User's main concern: [identify what they're expressing]
- Emotional content: [what emotions/feelings are present]
- Intensity level: [0=neutral, 1-2=mild, 3-4=moderate, 5=high emotional intensity]
- Response approach: [how to respond empathetically]
</reasoning>
<answer>
[Your empathetic response here - 1-2 sentences maximum]
</answer>""".strip()


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
