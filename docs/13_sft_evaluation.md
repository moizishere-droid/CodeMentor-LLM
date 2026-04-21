# Phase 13 — SFT Evaluation

## Goal
Evaluate fine-tuned Llama-3.2-3B-Instruct on test split.
Compare base model vs SFT model using ROUGE and BERTScore.

## Evaluation Config
- Model         : Abdulmoiz123/codementor-llm-sft
- Base Model    : meta-llama/Llama-3.2-3B-Instruct
- Test samples  : 50
- Metrics       : ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

## ROUGE Scores
| Metric  | Base   | SFT    | Improvement |
|---------|--------|--------|-------------|
| ROUGE-1 | 0.2258 | 0.2585 | +0.0327     |
| ROUGE-2 | 0.1072 | 0.1545 | +0.0473     |
| ROUGE-L | 0.1749 | 0.2256 | +0.0507     |

## BERTScore
| Metric    | Base   | SFT    | Improvement |
|-----------|--------|--------|-------------|
| Precision | 0.8037 | 0.7989 | -0.0048     |
| Recall    | 0.8736 | 0.8798 | +0.0061     |
| F1        | 0.8366 | 0.8363 | -0.0003     |

## Qualitative Analysis

### SFT Model Strengths
- Concise direct code answers
- Correct syntax for simple tasks
- ROUGE improved across all metrics
- Faster to the point than base model

### SFT Model Weaknesses
- Repetitive output loops (Q1, Q4, Q5)
- Hallucinating outputs on complex tasks
- Base model better for explanations (Q7, Q8)

### Root Cause of Issues
- Only 5K training samples — not enough for complex tasks
- Needs repetition_penalty in inference config
- More data or epochs would fix repetition issues

## Key Findings
- ROUGE-L improved by +0.05 — significant for coding dataset
- BERTScore nearly identical — semantic understanding unchanged
- SFT model better for simple coding tasks
- DPO alignment in Phase 15 will improve response quality

## Results Storage
- data/results/sft_evaluation_results.json

## Source File
- backend/src/evaluate.py