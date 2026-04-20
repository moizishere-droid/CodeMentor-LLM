# Phase 05 — Data Formatting & Prompt Template Design

## Goal
Convert CodeAlpaca-20K dataset to Llama-3 chat template format
for supervised fine-tuning (SFT).

## Llama-3 Chat Template Format
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>

## System Prompt
"You are a helpful coding assistant. Answer coding questions
clearly and concisely with working code examples."

## Format Logic
- If sample has input field — combine instruction + input as user message
- If sample has no input field — use instruction only as user message
- Always include system prompt
- Always include assistant output

## Token Length Analysis
- Min tokens  : 66
- Max tokens  : 1194
- Mean tokens : 136.79
- Median      : 122.0
- Samples under 2048 tokens : 20,022 (100%)
- Samples over 2048 tokens  : 0 (0%)
- All samples within Llama-3 context window — no filtering needed

## Dataset Storage
- Formatted dataset: Abdulmoiz123/codementor-llm-formatted (HuggingFace Hub)
- Locally: data/processed/
- Total samples: 20,022
- Column: text (single column containing full formatted prompt)

## Source File
- backend/src/data_formatter.py

## Key Decisions
- Used Llama-3 official chat template via tokenizer.apply_chat_template()
- Combined instruction + input fields when input exists
- Removed original columns — kept only text column
- All 20,022 samples formatted successfully