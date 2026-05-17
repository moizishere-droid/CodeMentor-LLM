# Phase 21 — Streamlit Frontend

## Goal
Build simple single-page Streamlit frontend
that calls the FastAPI backend.

## Frontend Structure
frontend/
├── app.py           ← single page Streamlit app
├── requirements.txt ← streamlit, requests
└── Dockerfile       ← frontend container

## Features
- Text area for coding question input
- Max tokens slider (64-1024)
- Submit button
- Response displayed in markdown
- Inference latency displayed
- Error handling for API connection issues

## API Integration
- Method  : POST
- URL     : http://localhost:8000/generate
- Payload : prompt + max_new_tokens
- Timeout : 60 seconds

## Error Handling
- Empty input warning
- API connection error
- Request timeout
- Unexpected errors

## Key Design Decisions
- Single page — no sidebar, no history
- Minimal UI — focus on functionality
- Markdown rendering — code blocks formatted properly
- Environment variable for API URL — easy to change for deployment

## Running Locally
```bash
cd frontend
streamlit run app.py
```