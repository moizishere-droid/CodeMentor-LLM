"""
API Routes for CodeMentor-LLM
Defines all API endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse, LogsResponse
from api.database import get_db, log_inference, InferenceLog
from api.model_loader import generate_response

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model="Abdulmoiz123/codementor-llm-merged",
        version="1.0.0"
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, db: Session = Depends(get_db)):
    """Generate response for a coding prompt."""
    # Generate response via HF Inference API
    result = generate_response(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    # Log to database
    log_inference(
        db=db,
        prompt=request.prompt,
        response=result["response"],
        latency_ms=result["latency_ms"],
        success=result["success"]
    )

    return GenerateResponse(
        response=result["response"],
        latency_ms=result["latency_ms"],
        success=result["success"]
    )


@router.get("/logs", response_model=LogsResponse)
def get_logs(limit: int = 10, db: Session = Depends(get_db)):
    """Retrieve recent inference logs."""
    logs = db.query(InferenceLog)\
             .order_by(InferenceLog.timestamp.desc())\
             .limit(limit)\
             .all()
    total = db.query(InferenceLog).count()
    return LogsResponse(logs=logs, total=total)