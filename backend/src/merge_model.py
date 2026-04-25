"""
Model Merging Script for CodeMentor-LLM
Merges DPO LoRA adapter into base model weights
for deployment-ready inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import MODEL_ID, bnb_config


def load_model_with_adapter(model_id: str, adapter_id: str):
    """
    Load base model with LoRA adapter.
    Args:
        model_id: HuggingFace base model ID
        adapter_id: HuggingFace adapter ID
    Returns:
        tuple of model and tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_id)
    print(f"Model loaded — {model.get_memory_footprint() / 1024**3:.2f} GB")

    return model, tokenizer


def merge_adapter(model) -> AutoModelForCausalLM:
    """
    Merge LoRA adapter into base model weights.
    Args:
        model: PeftModel with LoRA adapter
    Returns:
        Merged LlamaForCausalLM model
    """
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    print(f"Merged model type: {type(merged_model)}")
    print(f"Memory footprint: {merged_model.get_memory_footprint() / 1024**3:.2f} GB")
    return merged_model


def push_merged_model(merged_model, tokenizer, hub_model_id: str):
    """
    Push merged model to HuggingFace Hub.
    Args:
        merged_model: merged LlamaForCausalLM model
        tokenizer: tokenizer
        hub_model_id: HuggingFace Hub model ID
    """
    print(f"Pushing merged model to {hub_model_id}...")
    merged_model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    print(f"Model available at: https://huggingface.co/{hub_model_id}")


if __name__ == "__main__":
    # Load DPO model
    model, tokenizer = load_model_with_adapter(
        MODEL_ID,
        "Abdulmoiz123/codementor-llm-dpo"
    )

    # Merge adapter
    merged_model = merge_adapter(model)

    # Push to HF Hub
    push_merged_model(
        merged_model,
        tokenizer,
        "Abdulmoiz123/codementor-llm-merged"
    )