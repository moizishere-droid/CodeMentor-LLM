# Phase 11 — Stage 1: SFT Training with SFTTrainer

## Goal
Fine-tune meta-llama/Llama-3.2-3B-Instruct on CodeAlpaca-20K
dataset using QLoRA + SFTTrainer for coding Q&A task.

## Training Config
- Model             : meta-llama/Llama-3.2-3B-Instruct
- Dataset           : Abdulmoiz123/codementor-llm-splits
- Train samples     : 5,000 (reduced from 15,219 due to T4 constraints)
- Val samples       : 1000
- Epochs            : 3
- Batch size        : 4
- Gradient accum    : 4
- Effective batch   : 16
- Learning rate     : 2e-4
- Warmup ratio      : 0.05
- Optimizer         : paged_adamw_32bit
- LR Scheduler      : cosine
- Max seq length    : 2048
- Quantization      : QLoRA 4-bit NF4
- LoRA rank         : 16
- LoRA alpha        : 32

## Training Results
| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 0.5079     | 0.5381   |
| 2     | 0.4660     | 0.5340   |
| 3     | completed  | —        |

## Key Observations
- Loss decreasing consistently — model is learning
- No overfitting — val loss closely follows train loss
- Training time: 1h 54min on free T4 GPU
- Adapter size: 36.7MB — very lightweight

## Limitations
- Trained on 5K samples due to free T4 GPU constraints
- Full dataset (15,219 samples) would improve performance
- 3 epochs sufficient for this dataset size

## Model Storage
- HuggingFace Hub: Abdulmoiz123/codementor-llm-sft
- Adapter: adapter_model.safetensors (36.7MB)
- Config: adapter_config.json
- Tokenizer: tokenizer.json

## Experiment Tracking
- Platform: Weights & Biases
- Project: codementor-llm
- Run: sft-llama3.2-3b-codealapaca

## Source File
- backend/src/train_sft.py