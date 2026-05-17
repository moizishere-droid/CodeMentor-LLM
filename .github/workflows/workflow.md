# CI workflow

Push code
    ↓
Job 1: ruff lint passes? ✅
    ↓
Job 2: pytest 29 tests pass? ✅
    ↓
Green checkmark on GitHub commit ✅

# Why CI Only ?

We only do CI (Continuous Integration) — automated testing and linting.
CD (Continuous Deployment) would automatically deploy to HF Spaces after tests pass. We are not doing CD because:

1) HF Spaces deploys automatically when you push to its git repo (separate from GitHub)
2) Adding CD would mean auto-pushing to HF Spaces on every commit — risky for a portfolio project
3) Manual deployment is safer and more controlled