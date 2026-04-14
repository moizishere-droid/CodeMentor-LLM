# Phase 10 — LoRA Adapter Configuration

## Goal
Apply LoRA adapter to quantized meta-llama/Meta-Llama-3-8B-Instruct
for parameter efficient fine-tuning.

## LoRA Config
- r (rank)       : 16
- lora_alpha     : 32
- target_modules : q_proj, k_proj, v_proj, o_proj
- lora_dropout   : 0.05
- bias           : none
- task_type      : CAUSAL_LM

## Trainable Parameters
- Trainable params : 13,631,488 (13.6M)
- Total params     : 8,043,892,736 (8.04B)
- Trainable %      : 0.17%
- Frozen params    : 99.83% (base model weights)

## Why These Target Modules
- q_proj — query projection — learns what to attend to
- k_proj — key projection — learns what to compare against
- v_proj — value projection — learns what information to extract
- o_proj — output projection — learns how to combine attention

## Why r=16 and alpha=32
- r=16 — good balance between capacity and efficiency
- alpha=32 — scaling factor = alpha/r = 2.0
- Higher alpha = stronger LoRA influence on model output
- r=16 is standard for 7-8B models on T4 GPU

## Checkpoint Test
- adapter_model.safetensors — LoRA adapter weights
- adapter_config.json — LoRA configuration
- Only adapter files saved — not full 16GB model

## QLoRA Summary
- Base model frozen in 4-bit (5.21 GB)
- LoRA adapters trainable (~100 MB)
- Total memory: 7.17 GB
- Free for training: 7.39 GB

## Source File
- backend/src/config.py