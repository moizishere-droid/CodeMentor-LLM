# Phase 01 — Problem Framing & Project Scoping

## Project Name
CodeMentor-LLM

## Base Model
Base Model: meta-llama/Meta-Llama-3-8B-Instruct (already smart, but generic)

## Task
General Coding Q&A — explain code, debug errors, write code snippets

## Problem Statement
Developers face three daily problems:
- Don't understand why their code is broken
- Can't find clear explanation for a concept
- Need working code snippets fast

General purpose LLMs are not fine-tuned for structured,
instructional, focused coding responses.

## Two Stage Training Plan
- Stage 1 — SFT with QLoRA 4-bit on CodeAlpaca-20K --> “learn how to answer”
- Stage 2 — DPO with LoRA on preference dataset --> “learn how to answer better”

## Success Metrics
- ROUGE-L improvement over base model
- BERTScore improvement on test set
- Model correctly debugs 8/10 sample error prompts
- API response latency under 10 seconds
- Full pipeline reproducible from docker-compose up

## Datasets
- Primary: sahil2801/CodeAlpaca-20k (Apache 2.0)
- DPO: constructed from SFT model outputs
- Backup: iamtarun/python_code_instructions_18k_alpaca

## Scope
### In Scope
- Python, JavaScript, general programming
- Explain code, debug errors, write functions
- REST API + Streamlit frontend
- Docker deployment

### Out of Scope
- Real-time code execution
- Multi-turn memory
- Fine-tuning on proprietary codebases