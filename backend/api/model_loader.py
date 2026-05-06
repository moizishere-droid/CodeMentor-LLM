"""
Model Loader for CodeMentor-LLM API
Loads merged model once at startup and keeps it in memory.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Global model and tokenizer
model = None
tokenizer = None

MERGED_MODEL_ID = "Abdulmoiz123/codementor-llm-merged"


def load_model():
    """
    Load merged model and tokenizer at startup.
    Called once via FastAPI lifespan.
    """
    global model, tokenizer

    print(f"Loading model from {MERGED_MODEL_ID}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Model loaded successfully — {model.get_memory_footprint() / 1024**3:.2f} GB")


def get_model():
    """Return loaded model."""
    return model


def get_tokenizer():
    """Return loaded tokenizer."""
    return tokenizer