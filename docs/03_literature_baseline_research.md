# Phase 03 — Literature & Baseline Research

## Model Tested
- Model: meta-llama/Meta-Llama-3-8B-Instruct
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
- Answers all 10 questions correctly
- Formats code blocks consistently with backticks
- Gives step-by-step explanations (Q4, Q7)
- Provides multiple solution approaches (Q1 — slice + reversed)
- Uses bold headers to structure answers (Q3, Q8)
- Handles Python, SQL, and conceptual questions
- Good real-world analogies (Q10 — restaurant analogy)
- Proper error handling examples (Q5 — try/except/else)

## Weaknesses:
- Responses cut off mid-sentence in 6/10 questions (Q2, Q3, Q4, Q7, Q8, Q9)
- No consistent response length
- Attention mask warning on every inference
- No domain-specific coding instruction format
- Sometimes over-explains simple concepts
- Q8 has wrong example — `[1,2,3] == [1,2,3]` returns `True` not `False`
- Q10 wrong full form — "Representational State of Mind" instead of "Representational State Transfer"

## Key Issues Justifying Fine-Tuning:
1. Responses cut off in 60% of questions — major quality issue
2. Factual errors present (Q8, Q10)
3. No consistent teaching structure
4. SFT will teach complete structured responses
5. DPO will align model to prefer accurate higher quality responses

## Model Decision
- Final confirmed model: meta-llama/Meta-Llama-3-8B-Instruct
- Access: Granted
- All training phases will use this model