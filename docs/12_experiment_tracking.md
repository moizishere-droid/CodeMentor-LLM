# Phase 12 — SFT Experiment Tracking & Checkpoint Selection

## W&B Run
- Project  : codementor-llm
- Run name : sft-llama3.2-3b-codealapaca
- URL      : https://wandb.ai/abdulmoiz2004-2-institution-of-engineering-and-technology/codementor-llm

## Training Results
| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1     | 0.5079     | 0.5381   | 0.8505       |
| 2     | 0.4660     | 0.5320   | 0.8530       |
| 3     | ~0.45      | 0.5340   | 0.8535       |

## Curve Analysis
- Train loss    : dropped sharply 2.5 → 0.5 — strong learning
- Train accuracy: rose consistently 0.60 → 0.85
- Eval loss     : decreased 0.538 → 0.532 — no overfitting
- Eval accuracy : increasing 0.850 → 0.853
- LR schedule   : cosine decay working correctly
- Grad norm     : stabilized at 0.25-0.5 after warmup

## Best Checkpoint
- Epoch         : 2
- Val loss      : 0.532 (lowest)
- Val accuracy  : 0.853
- Location      : Abdulmoiz123/codementor-llm-sft
- Step          : ~624

## Key Decisions
- Epoch 2 selected as best checkpoint
- No overfitting observed
- Training on 5K samples sufficient for meaningful learning
- Cosine LR schedule + paged_adamw_32bit optimizer worked well