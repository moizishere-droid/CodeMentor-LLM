# Phase 15 — Stage 2: DPO Training with LoRA Adapter

## Goal
Align fine-tuned Llama-3.2-3B-Instruct using Direct
Preference Optimization (DPO) on preference dataset.

## Training Config
- Base model  : meta-llama/Llama-3.2-3B-Instruct
- SFT adapter : Abdulmoiz123/codementor-llm-sft
- Dataset     : Abdulmoiz123/codementor-llm-preference
- Beta        : 0.1
- Epochs      : 1
- LR          : 5e-5
- Batch size  : 4
- Samples     : 200
- Optimizer   : paged_adamw_32bit
- Scheduler   : cosine

## Training Results
- Training loss : 0.0056
- Steps         : 12
- Training time : ~1.5 minutes

## Key Decisions
- Beta=0.1 — standard DPO beta value
- 1 epoch — DPO overfits quickly
- 200 pairs — limited by T4 GPU time
- Low loss (0.0056) — model learned preference quickly

## W&B Run
- Project : codementor-llm
- Run     : dpo-llama3-3b
- URL     : https://wandb.ai/abdulmoiz2004-2-institution-of-engineering-and-technology/codementor-llm/runs/pjp882g7

## Model Storage
- HuggingFace Hub: Abdulmoiz123/codementor-llm-dpo

## Source File
- backend/src/train_dpo.py