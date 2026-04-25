# Phase 18 — Inference Pipeline

## Goal
Build production-ready inference pipeline
for the merged fine-tuned model.

## Pipeline Flow
1. User input → validate input
2. Apply Llama-3 chat template
3. Tokenize input
4. Model generate
5. Decode output
6. Return response + latency

## Inference Config
- Model              : Abdulmoiz123/codementor-llm-merged
- max_new_tokens     : 512
- temperature        : 0.7
- top_p              : 0.9
- do_sample          : True
- repetition_penalty : 1.3

## Edge Cases Handled
- Empty input
- Whitespace only input
- Non-string input
- Input too long (> 2048 chars)
- Model generation error

## Response Format
```json
{
  "response": "generated response text",
  "latency_ms": 1234.56,
  "success": true
}
```

## Expected Latency
- T4 GPU  : 2-5 seconds

## Source File
- backend/src/inference.py