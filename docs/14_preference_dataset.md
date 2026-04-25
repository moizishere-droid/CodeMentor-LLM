# Phase 14 — Preference Dataset Construction

## Goal
Build preference dataset for DPO training.
Chosen = SFT model response
Rejected = Base model response

## Dataset Construction Method
- Source: Abdulmoiz123/codementor-llm-splits (train split)
- Samples used: 200
- For each sample:
  - Enable SFT adapter → generate chosen response
  - Disable SFT adapter → generate base response
  - Format as prompt/chosen/rejected pair

## Generation Config
- max_new_tokens : 128
- temperature    : 0.7
- top_p          : 0.9
- repetition_penalty: 1.3

## Dataset Format
```json
{
  "prompt": "instruction text",
  "chosen": "SFT model response",
  "rejected": "base model response"
}
```

## Observations
- SFT responses: concise, direct code, less explanation
- Base responses: verbose, well structured, better explanations
- DPO will learn style differences and improve quality

## Dataset Storage
- HuggingFace Hub: Abdulmoiz123/codementor-llm-preference
- Local: data/processed/preference_dataset.jsonl
- Total pairs: 200

## Key Decisions
- 200 pairs due to T4 GPU time constraints
- Industry standard is 1K-10K pairs
- 200 pairs sufficient for portfolio demonstration
- max_new_tokens=128 to speed up generation