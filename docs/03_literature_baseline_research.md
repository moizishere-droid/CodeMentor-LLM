# Phase 03 — Literature & Baseline Research

## Model Tested
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Quantization: 4-bit QLoRA (NF4)
- GPU: Tesla T4 — 14.56GB
- Memory footprint after loading: 3.75GB

## Baseline Test
- 10 coding questions tested against base model
- No fine-tuning, no system prompt
- Results saved to data/results/baseline_results.json

## Strengths of Base Model
- Answers basic coding questions correctly
- Formats code blocks properly
- Gives multiple solution approaches
- Explains concepts clearly

## Weaknesses of Base Model
- Responses cut off mid-sentence in 4/10 questions
- No consistent response structure
- No step-by-step teaching format
- Inconsistent response length
- Attention mask warning during inference

## Fine-Tuning Justification
Fine-tuning on CodeAlpaca-20K is justified because:
1. Base model lacks consistent response structure
2. Responses cut off frequently
3. No domain-specific coding instruction format
4. SFT will teach structured complete responses
5. DPO will align model to prefer higher quality responses

## Model Decision
- Primary model: meta-llama/Meta-Llama-3-8B-Instruct
- Status: Access requested — waiting for approval
- Temporary baseline: mistralai/Mistral-7B-Instruct-v0.3
- Baseline testing done on Mistral-7B until Llama-3 access granted

## Note
Once Llama-3-8B-Instruct access is approved:
- Re-run baseline test on Llama-3
- Update baseline_results.json
- All training phases will use Llama-3-8B-Instruct