"""
Configuration for CodeMentor-LLM
Contains quantization, LoRA, training and inference configs.
"""

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType


# Model Config
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HUB_MODEL_ID = "Abdulmoiz123/codementor-llm"

# Quantization Config (QLoRA 4-bit NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Config (filled in Phase 10)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# SFT Training Config (filled in Phase 11)
SFT_CONFIG = {
    "dataset_id": "Abdulmoiz123/codementor-llm-splits",
    "output_dir": "./checkpoints/sft",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "fp16": True,
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "max_seq_length": 2048,
    "report_to": "wandb",
}

# DPO Training Config (filled in Phase 15)
DPO_CONFIG = {
    "output_dir": "./checkpoints/dpo",
    "beta": 0.1,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "fp16": True,
    "logging_steps": 10,
    "report_to": "wandb",
}

# Inference Config
INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# System Prompt
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)