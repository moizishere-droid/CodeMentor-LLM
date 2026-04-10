"""
Data Formatter for CodeMentor-LLM
Converts CodeAlpaca dataset samples to Llama-3 chat template format.
"""

from transformers import AutoTokenizer

# System prompt for coding assistant
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)


def load_tokenizer(model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load Llama-3 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


def format_prompt(sample: dict, tokenizer) -> dict:
    """
    Converts a CodeAlpaca sample to Llama-3 chat template format.
    Args:
        sample: dict with keys instruction, input, output
        tokenizer: Llama-3 tokenizer
    Returns:
        dict with key text containing formatted prompt
    """
    # Combine instruction and input if input exists
    if sample["input"] and sample["input"].strip():
        user_message = f"{sample['instruction']}\n\nInput:\n{sample['input']}"
    else:
        user_message = sample["instruction"]

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": sample["output"]}
    ]

    # Apply Llama-3 chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": formatted}


def format_dataset(dataset, tokenizer):
    """
    Apply format_prompt to entire dataset.
    Args:
        dataset: HuggingFace dataset
        tokenizer: Llama-3 tokenizer
    Returns:
        Formatted dataset with single text column
    """
    formatted = dataset.map(
        lambda sample: format_prompt(sample, tokenizer),
        remove_columns=dataset.column_names
    )
    return formatted


if __name__ == "__main__":
    from datasets import load_dataset

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load dataset
    dataset = load_dataset("sahil2801/CodeAlpaca-20k")

    # Format dataset
    formatted = format_dataset(dataset["train"], tokenizer)

    print(f"Formatted dataset size: {len(formatted)}")
    print(f"Sample:\n{formatted[0]['text']}")