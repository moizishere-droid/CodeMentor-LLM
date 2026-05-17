# Phase 22 — Testing (pytest)

## Goal
Write unit tests for all critical functions using pytest.

## Test Files
- tests/test_data_formatter.py — data formatting functions
- tests/test_data_cleaner.py   — data cleaning functions
- tests/test_inference.py      — inference pipeline
- tests/test_api.py            — API endpoints

## Test Results
- Total tests  : 29
- Passed       : 29
- Failed       : 0
- Warnings     : 3 (deprecation warnings — non critical)

## Test Coverage

### test_data_formatter.py
- format_preference_pair basic
- format_preference_pair keys
- format_preference_pair empty strings
- system prompt exists
- format_preference_pair returns dict

### test_data_cleaner.py
- remove_duplicates basic
- remove_duplicates no duplicates
- remove_nulls basic
- remove_nulls no nulls
- is_low_quality short response
- is_low_quality good response
- filter_low_quality basic
- remove_duplicates returns tuple
- remove_nulls empty string

### test_inference.py
- validate_input valid
- validate_input empty
- validate_input whitespace
- validate_input too long
- validate_input non string
- validate_input normal question
- validate_input max length
- validate_input returns tuple

### test_api.py
- health endpoint
- generate endpoint valid
- generate endpoint empty prompt
- generate endpoint missing prompt
- logs endpoint
- logs endpoint with limit
- generate endpoint max tokens limit

## Running Tests
```bash
cd backend
pytest tests/ -v
```