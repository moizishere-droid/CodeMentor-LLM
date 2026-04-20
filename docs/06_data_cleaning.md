# Phase 06 — Data Cleaning & Quality Filtering

## Goal
Clean and filter the formatted CodeAlpaca-20K dataset
to remove low quality samples before training.

## Cleaning Pipeline

### Step 1 — Null & Empty Check
- Null values found: 0
- Empty text fields: 0
- Samples removed: 0

### Step 2 — Deduplication
- Duplicate rows found: 0
- Samples removed: 0

### Step 3 — Token Length Filter
- Min tokens: 66
- Max tokens: 1194
- Mean tokens: 136.79
- Filter range: 10 to 2048 tokens
- Samples removed: 0

### Step 4 — Low Quality Filter
- Rule: assistant response shorter than 3 words
- Low quality samples found: 998
- Samples removed: 998

## Cleaning Summary
- Original samples        : 20,022
- After deduplication     : 20,022
- After token filter      : 20,022
- After quality filter    : 19,024
- Total removed           : 998
- Total remaining         : 19,024
- Retention rate          : 95.02%

## Dataset Storage
- Cleaned dataset: Abdulmoiz123/codementor-llm-cleaned (HuggingFace Hub)
- Local: data/processed/
- Total samples: 19,024
- Column: text

## Source File
- backend/src/data_cleaner.py

## Key Decisions
- Removed samples with assistant response shorter than 3 words
- All samples within 2048 token limit — no filtering needed
- No duplicates found in CodeAlpaca-20K dataset
- 95.02% retention rate — very healthy clean dataset