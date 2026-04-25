"""
DPO Training Script for CodeMentor-LLM
Aligns fine-tuned Llama-3.2-3B-Instruct using
Direct Preference Optimization (DPO).
"""

import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

from config import (
    MODEL_ID,
    bnb_config,
    DPO_CONFIG,
)


def load_tokenizer(model_id: str):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_sft_model(model_id: str, sft_adapter_id: str):
    """Load base model with SFT adapter for DPO training."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_id,
        is_trainable=True,
    )
    print(f"Model loaded — {model.get_memory_footprint() / 1024**3:.2f} GB")
    return model


def load_preference_dataset():
    """Load preference dataset from HuggingFace Hub."""
    dataset = load_dataset("Abdulmoiz123/codementor-llm-preference")
    print(f"Preference dataset: {len(dataset['train'])} samples")
    return dataset


def train():
    """Main DPO training function."""
    # Initialize W&B
    wandb.init(
        project="codementor-llm",
        name="dpo-llama3-3b",
        config=DPO_CONFIG
    )

    # Load tokenizer and model
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_sft_model(MODEL_ID, "Abdulmoiz123/codementor-llm-sft")

    # Load preference dataset
    dataset = load_preference_dataset()

    # DPO Config
    dpo_config = DPOConfig(
        output_dir=DPO_CONFIG["output_dir"],
        num_train_epochs=DPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=DPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=DPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=DPO_CONFIG["learning_rate"],
        fp16=DPO_CONFIG["fp16"],
        logging_steps=DPO_CONFIG["logging_steps"],
        save_strategy="epoch",
        report_to="wandb",
        push_to_hub=True,
        hub_model_id="Abdulmoiz123/codementor-llm-dpo",
        hub_strategy="every_save",
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        beta=DPO_CONFIG["beta"],
        max_length=512,
        max_prompt_length=256,
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Push final model
    trainer.push_to_hub()

    # Finish W&B
    wandb.finish()

    print("DPO Training complete")
    print("Model: https://huggingface.co/Abdulmoiz123/codementor-llm-dpo")


if __name__ == "__main__":
    train()