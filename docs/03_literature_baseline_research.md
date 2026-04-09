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

## Questions Tested
1. Write a Python function to reverse a string
2. Explain what a decorator is in Python with an example
3. What is the difference between a list and a tuple in Python
4. Write a Python function to check if a number is prime
5. How do you handle exceptions in Python
6. Write a SQL query to find the second highest salary
7. Explain the concept of recursion with a Python example
8. What is the difference between == and is in Python
9. Write a Python function to find duplicates in a list
10. Explain what a REST API is in simple terms

## Strengths of Base Model
- Answers all 10 coding questions correctly
- Formats code blocks consistently
- Gives clear step-by-step explanations
- Handles multiple question types — Python, SQL, concepts
- Uses bold headers to structure answers
- Provides real world analogies for concepts
- Better code quality compared to Mistral-7B baseline

## Weaknesses of Base Model
- Responses cut off mid-sentence in 5/10 questions
- No consistent response length
- Attention mask warning during inference
- No specific coding instruction format
- Sometimes over-explains simple concepts
- SQL syntax not always compatible across databases

## Fine-Tuning Justification
1. Responses cut off frequently
2. No consistent coding instruction format
3. SFT on CodeAlpaca-20K will teach structured complete responses
4. DPO will align model to prefer higher quality responses
5. Fine-tuned model will be domain-specific and focused

## Model Decision
- Final confirmed model: meta-llama/Meta-Llama-3-8B-Instruct
- Access: Granted
- All training phases will use this model