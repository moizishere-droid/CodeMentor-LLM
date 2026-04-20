# Phase 03 — Literature & Baseline Research

## Model Tested
- Model: meta-llama/Llama-3.2-3B-Instruct
- Quantization: 4-bit QLoRA (NF4)
- GPU: Tesla T4 — 14.56GB
- Memory footprint after loading: 5.21GB

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

## Strengths:
- Answers all 10 coding questions correctly
- Formats code blocks consistently with backticks
- Includes docstrings in code (Q1, Q4, Q9) — good practice
- Uses bold headers and structured sections (Q2, Q4, Q6)
- Provides real-world analogies (Q10 — restaurant analogy)
- SQL answer uses DENSE_RANK() — more accurate than 8B version
- Correct identity operator examples (Q8)
- Handles Python, SQL, and conceptual questions well

## Weaknesses:
- Responses cut off mid-sentence in 5/10 questions (Q2, Q3, Q4, Q7, Q8)
- No consistent response length
- Attention mask warning on every inference
- No specific coding instruction format
- Q10 wrong full form — "Representational State of Resource" instead of "Representational State Transfer"
- Over-explains simple concepts (Q9)
- Inconsistent output quality compared to larger models

### Why Fine-Tuning is justified:
1. Responses cut off in 50% of questions
2. Factual errors present (Q10 wrong full form)
3. No consistent coding instruction format
4. SFT on CodeAlpaca-20K will teach complete structured responses
5. DPO will align model to prefer accurate higher quality responses
6. Fine-tuned model will be domain-specific and focused

## Model Decision
- Final confirmed model: meta-llama/Meta-Llama-3-8B-Instruct
- Access: Granted
- All training phases will use this model