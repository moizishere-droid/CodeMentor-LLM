# Phase 08 — Train / Val / Test Split & Dataset Versioning

## Split Strategy
- Train : 80%
- Val   : 10%
- Test  : 10%
- Seed  : 42 (fixed for reproducibility)

## Split Results
- Train      : 15,219 samples (80%)
- Validation :  1,902 samples (10%)
- Test       :  1,903 samples (10%)
- Total      : 19,024 samples

## Split Method
- Step 1: Split full dataset into 80% train and 20% temp
- Step 2: Split 20% temp into 10% val and 10% test
- Used HuggingFace train_test_split() with seed=42

## Dataset Storage
- HuggingFace Hub: Abdulmoiz123/codementor-llm-splits
- Local:
  - data/processed/train.jsonl
  - data/processed/validation.jsonl
  - data/processed/test.jsonl

## Key Decisions
- Fixed seed 42 for reproducibility
- Stratified split not needed — single text column
- All splits pushed to HuggingFace Hub for versioning
- Test split will be used for final evaluation only