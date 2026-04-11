# Phase 07 — Exploratory Data Analysis (EDA)

## Dataset
- Source: Abdulmoiz123/codementor-llm-cleaned
- Total samples: 19,024

## Token Length Analysis
- Min    : 48
- Max    : 1,174
- Mean   : 119.42
- Median : 105.0
- Std    : 58.76
- All samples within 2048 token limit

## Response Word Count Analysis
- Min    : 3
- Max    : 479
- Mean   : 27.72
- Median : 18.0
- Most responses are short (10-50 words) — typical for coding answers

## Top 20 Keywords
1. input     — 10,576
2. create    — 5,058
3. write     — 4,881
4. code      — 3,327
5. string    — 3,291
6. function  — 3,136
7. list      — 3,123
8. array     — 3,065
9. python    — 2,612
10. numbers  — 2,413
11. name     — 2,353
12. number   — 2,250
13. all      — 2,114
14. following — 2,101
15. two      — 1,975
16. program  — 1,833
17. table    — 1,591
18. query    — 1,577
19. javascript — 1,540
20. find     — 1,496

## Topic Balance Analysis
- write_code : 11,461 (60.24%)
- other      :  6,303 (33.13%)
- explain    :    926 (4.87%)
- debug      :    270 (1.42%)
- optimize   :     64 (0.34%)

## Key Findings
- Dataset is heavily write_code focused (60%) — good for our use case
- Debug (1.42%) and explain (4.87%) are underrepresented
- Right-skewed token distribution — most samples are short
- Python and JavaScript are the dominant languages
- All samples within 2048 token limit — no further filtering needed

## Plots
- data/results/token_length_distribution.png
- data/results/response_length_distribution.png
- data/results/topic_balance.png
- data\results\top_20_keywords.png