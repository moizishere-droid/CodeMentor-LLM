# Phase 23 — CI/CD with GitHub Actions

## Goal
Automate code quality checks and testing on every
push to main branch using GitHub Actions.

## Why CI/CD
- Every commit automatically verified
- Catches bugs before they reach production
- Shows professional engineering discipline
- Green checkmarks on GitHub commits

## Pipeline Structure
Push code to GitHub
↓
Job 1 — Lint (ruff)
↓ passes
Job 2 — Test (pytest)
↓ passes
Green checkmark 

## CI Pipeline File
- Location: .github/workflows/ci.yml
- Trigger: push or pull request to main branch

## Job 1 — Lint
- Tool: ruff
- Checks: backend/src and backend/api
- Catches: syntax errors, unused imports, bad style
- Ignored: E501 (long lines)

## Job 2 — Test
- Tool: pytest
- Tests: backend/tests/
- Total: 29 tests
- Runs only if lint passes

## Test Coverage
- test_data_formatter.py — 5 tests
- test_data_cleaner.py   — 9 tests
- test_inference.py      — 8 tests
- test_api.py            — 7 tests

## Secrets Used
- HF_TOKEN — HuggingFace token for API calls
- MODAL_ENDPOINT — Modal serverless GPU endpoint

## What is NOT included
- CD (Continuous Deployment) — HF Spaces deploys automatically from its own git repo
- Docker build in CI — no GPU available on GitHub runners