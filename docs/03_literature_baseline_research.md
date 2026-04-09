# Phase 03 — Literature & Baseline Research

## Model Tested
- Model: mistralai/Mistral-7B-Instruct-v0.3
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

## Model Change
- Originally planned: meta-llama/Llama-3-8B-Instruct
- Changed to: mistralai/Mistral-7B-Instruct-v0.3
- Reason: Llama-3 is gated and requires access approval
- Mistral-7B is openly available and equally capable