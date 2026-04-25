# Phase 16 — DPO Evaluation & Full Comparison

## Goal
Three-way comparison of Base, SFT, and DPO models
on test split using ROUGE and BERTScore metrics.

## Evaluation Config
- Base  : meta-llama/Llama-3.2-3B-Instruct
- SFT   : Abdulmoiz123/codementor-llm-sft
- DPO   : Abdulmoiz123/codementor-llm-dpo
- Test samples: 50

## ROUGE Scores
| Metric  | Base   | SFT    | DPO    |
|---------|--------|--------|--------|
| ROUGE-1 | 0.2408 | 0.2639 | 0.2542 |
| ROUGE-2 | 0.0897 | 0.1017 | 0.1090 |
| ROUGE-L | 0.1807 | 0.2004 | 0.2077 |

## BERTScore
| Metric    | Base   | SFT    | DPO    |
|-----------|--------|--------|--------|
| Precision | 0.8132 | 0.8218 | 0.8208 |
| Recall    | 0.8664 | 0.8825 | 0.8808 |
| F1        | 0.8385 | 0.8504 | 0.8491 |

## Qualitative Analysis

### DPO Improvements over SFT
- Cleaner code structure
- Better function definitions
- More logical code flow
- Less repetitive output

### DPO Weaknesses
- Still cuts off mid-code
- Syntax errors present
- 200 preference pairs too small for major quality jump

### Overall Ranking
- Simple tasks  : DPO ≈ SFT > Base
- Complex tasks : Base > DPO ≈ SFT
- Code structure: DPO > SFT > Base

## Key Findings
- DPO best on ROUGE-2 and ROUGE-L — better sequence alignment
- BERTScore F1 — SFT slightly better than DPO
- Both SFT and DPO significantly better than base
- More preference pairs would improve DPO quality

## Results Storage
- data/results/dpo_evaluation_results.json