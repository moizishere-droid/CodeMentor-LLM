# Phase 09 — Quantization Config (QLoRA 4-bit NF4)

## Goal
Load meta-llama/Meta-Llama-3-8B-Instruct in 4-bit NF4 quantization
for memory efficient fine-tuning on T4 GPU.

## Quantization Config
- Type             : QLoRA 4-bit NF4
- Compute dtype    : bfloat16
- Double quant     : True
- Library          : bitsandbytes==0.45.3

## Memory Results
- GPU              : Tesla T4
- Total GPU Memory : 14.56 GB
- Original model   : ~16 GB (full precision)
- After 4-bit quant: 5.21 GB
- After k-bit prep : 7.17 GB
- Memory reduction : 68%
- Free for training: 7.39 GB

## Model Architecture
- q_proj   : Linear4bit(4096, 4096)
- k_proj   : Linear4bit(4096, 1024)
- v_proj   : Linear4bit(4096, 1024)
- o_proj   : Linear4bit(4096, 4096)
- gate_proj: Linear4bit(4096, 14336)
- up_proj  : Linear4bit(4096, 14336)
- down_proj: Linear4bit(14336, 4096)

## Target Modules for LoRA
- q_proj
- k_proj
- v_proj
- o_proj

## Key Decisions
- NF4 chosen over int4 — better quality for LLMs
- bfloat16 compute dtype — stable training on T4
- Double quantization — extra memory saving
- prepare_model_for_kbit_training() — enables gradient checkpointing

## Source File
- backend/src/config.py