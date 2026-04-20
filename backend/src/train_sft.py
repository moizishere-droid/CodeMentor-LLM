"""
SFT Training Script for CodeMentor-LLM
Fine-tunes Llama-3.2-3B-Instruct on CodeAlpaca dataset
using QLoRA + SFTTrainer.
"""

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

from config import (
    MODEL_ID,
    bnb_config,
    lora_config,
    SFT_CONFIG,
    SYSTEM_PROMPT
)


def load_tokenizer(model_id: str):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_id: str):
    """Load quantized model and prepare for training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_datasets():
    """Load train and validation splits."""
    dataset = load_dataset(SFT_CONFIG["dataset_id"])
    train_dataset = dataset["train"].select(range(5000))
    val_dataset = dataset["validation"].select(range(500))
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    return train_dataset, val_dataset


def train():
    """Main training function."""
    # Initialize W&B
    wandb.init(
        project="codementor-llm",
        name="sft-llama3.2-3b-codealapaca",
        config=SFT_CONFIG
    )

    # Load tokenizer and model
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID)

    # Load datasets
    train_dataset, val_dataset = load_datasets()

    # SFT Config
    sft_config = SFTConfig(
        output_dir=SFT_CONFIG["output_dir"],
        num_train_epochs=SFT_CONFIG["num_train_epochs"],
        per_device_train_batch_size=SFT_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
        learning_rate=SFT_CONFIG["learning_rate"],
        warmup_ratio=SFT_CONFIG["warmup_ratio"],
        fp16=SFT_CONFIG["fp16"],
        logging_steps=SFT_CONFIG["logging_steps"],
        eval_strategy=SFT_CONFIG["eval_strategy"],
        save_strategy=SFT_CONFIG["save_strategy"],
        load_best_model_at_end=True,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id="Abdulmoiz123/codementor-llm-sft",
        hub_strategy="every_save",
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_seq_length=SFT_CONFIG["max_seq_length"],
        dataset_text_field="text",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    # Train
    trainer.train()

    # Push final model
    trainer.push_to_hub()

    # Finish W&B
    wandb.finish()

    print("SFT Training complete")
    print(f"Model: https://huggingface.co/Abdulmoiz123/codementor-llm-sft")


if __name__ == "__main__":
    train()