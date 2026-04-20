# Phase 04 — Dataset Collection & Sourcing

## Datasets Collected

### Primary Dataset
- Name: CodeAlpaca-20K
- Source: sahil2801/CodeAlpaca-20k
- Size: 20,022 samples
- Columns: instruction, input, output
- License: Apache 2.0
- Format: Alpaca

### Backup Dataset
- Name: Python Code Instructions 18K
- Source: iamtarun/python_code_instructions_18k_alpaca
- Size: 18,612 samples
- Columns: instruction, input, output, prompt
- License: Apache 2.0
- Format: Alpaca

## Combined Stats
- Both datasets in Alpaca format
- Both compatible with Llama-3 chat template

## Sample Example — Primary Dataset
- Instruction: Create an array of length 5 which contains all even numbers between 1 and 10
- Input: (empty)
- Output: arr = [2, 4, 6, 8, 10]

## Sample Example — Backup Dataset
- Instruction: Create a function to calculate the sum of a sequence
- Input: [1, 2, 3, 4, 5]
- Output: Python function code

## Storage
- Primary raw dataset: Abdulmoiz123/codementor-llm-raw-primary (HuggingFace Hub)
- Backup raw dataset: Abdulmoiz123/codementor-llm-raw-backup (HuggingFace Hub)
- Local: data/raw/ (pulled from HF Hub when needed)

## Dataset Decision
- Primary dataset CodeAlpaca-20K will be used for SFT training
- Backup dataset will be used if more data is needed (not in this project right now)
- Both datasets are open source and free to use