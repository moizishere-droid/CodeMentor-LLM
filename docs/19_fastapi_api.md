# Phase 19 — FastAPI REST API + SQLite Logging

## Goal
Build production-ready REST API for model inference
with SQLite logging for every request.

## API Structure
backend/api/
├── main.py         ← FastAPI app + lifespan
├── routes.py       ← API endpoints
├── schemas.py      ← Pydantic request/response models
├── model_loader.py ← loads model once at startup
└── database.py     ← SQLite logging

## Endpoints
| Method | Endpoint   | Description                    |
|--------|------------|--------------------------------|
| GET    | /health    | API health check               |
| POST   | /generate  | Generate response for prompt   |
| GET    | /logs      | Retrieve recent inference logs |

## Request/Response Format

### POST /generate
Request:
- prompt        : string (1-2048 chars)
- max_new_tokens: int (1-1024, default 512)

Response:
- response  : generated text
- latency_ms: inference time in milliseconds
- success   : boolean

## Database Schema
Table: inference_logs
- id         : Integer (primary key)
- prompt     : String
- response   : String
- latency_ms : Float
- timestamp  : DateTime
- success    : Boolean

## Key Design Decisions
- Model loaded once at startup via lifespan — not per request
- SQLite chosen for simplicity — no external DB needed
- CORS enabled — allows frontend to call API
- Pydantic validation — rejects invalid requests automatically

## Tech Stack
- FastAPI    : REST API framework
- Pydantic   : request/response validation
- SQLAlchemy : database ORM
- SQLite     : lightweight database
- Uvicorn    : ASGI server