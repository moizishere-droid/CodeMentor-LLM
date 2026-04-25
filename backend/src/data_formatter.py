"""
Data Formatter for CodeMentor-LLM
Converts CodeAlpaca dataset samples to Llama-3 chat template format.
"""

from transformers import AutoTokenizer
from evaluate import extract_instruction, generate_response

# System prompt for coding assistant
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)

def load_tokenizer(model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
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

def format_preference_pair(prompt: str, chosen: str, rejected: str) -> dict:
    """
    Format a preference pair for DPO training.
    Args:
        prompt: instruction text
        chosen: preferred response (SFT model)
        rejected: non-preferred response (base model)
    Returns:
        dict with prompt, chosen, rejected keys
    """
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def build_preference_dataset(train_dataset, sft_model, tokenizer, n_samples: int = 200) -> list:
    """
    Build preference dataset from SFT and base model responses.
    Args:
        train_dataset: HuggingFace dataset
        sft_model: PeftModel with SFT adapter
        tokenizer: tokenizer
        n_samples: number of preference pairs to generate
    Returns:
        list of preference pairs
    """
    preference_pairs = []

    for i, sample in enumerate(train_dataset.select(range(n_samples))):
        instruction = extract_instruction(sample["text"])
        if not instruction:
            continue

        # Chosen — SFT model response
        sft_model.enable_adapter_layers()
        chosen = generate_response(sft_model, tokenizer, instruction)

        # Rejected — base model response
        sft_model.disable_adapter_layers()
        rejected = generate_response(sft_model, tokenizer, instruction)

        preference_pairs.append(
            format_preference_pair(instruction, chosen, rejected)
        )

    return preference_pairs

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load dataset
    dataset = load_dataset("sahil2801/CodeAlpaca-20k")

    # Format SFT dataset
    formatted = format_dataset(dataset["train"], tokenizer)
    print(f"Formatted dataset size: {len(formatted)}")

    # Build preference dataset
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
    )
    sft_model = PeftModel.from_pretrained(
        base_model,
        "Abdulmoiz123/codementor-llm-sft"
    )
    splits = load_dataset("Abdulmoiz123/codementor-llm-splits")
    pairs = build_preference_dataset(splits["train"], sft_model, tokenizer)
    print(f"Preference pairs: {len(pairs)}")