"""
API Routes for CodeMentor-LLM
Defines all API endpoints.
"""
"""
It is the bridge between:
1) Frontend / user request
2) Your AI model
3) Your database
"""

import json
import time
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.auth import current_user
from api.database import User
from api.schemas import GenerateRequest, GenerateResponse, HealthResponse, LogsResponse
from api.database import get_db, log_inference, InferenceLog, SessionLocal
from api.model_loader import generate_response, stream_generate_response

#from api.model_loader_prod import get_model, get_tokenizer --> for production, we load the model at startup and keep it in memory, so we don't need to get it for each request. Instead, we will directly call the generate_response function which uses the loaded model.
#from src.inference import generate_response --> for production, we will directly call the generate_response function from model_loader_prod which uses the loaded model, so we don't need to import it here. Instead, we will import it in main.py based on the environment variable.

router = APIRouter() # Create a router for API endpoints, allows us to organize routes and include them in the main app.


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model="Abdulmoiz123/codementor-llm-merged",
        version="1.0.0"
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """
    Generate response for a coding prompt.
    Args:
        request: GenerateRequest with prompt and max_new_tokens
        db     : database session
    Returns:
        GenerateResponse with response, latency_ms, success
    """
 
    # enable this for production when we load the model at startup and keep it in memory, so we don't need to get it for each request.
    # Instead, we will directly call the generate_response function which uses the loaded model.
    '''
    # Get model and tokenizer
    model = get_model()
    tokenizer = get_tokenizer()

    # Generate response
    result = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )
    '''
    
    result = generate_response(
    prompt=request.prompt,
    max_new_tokens=request.max_new_tokens,
    )

    # Log to database
    log_inference(
        db=db,
        user_id=user.id,
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


@router.post("/generate/stream")
def generate_stream(request: GenerateRequest, user: User = Depends(current_user)):
    """Stream generated text to the frontend using server-sent events."""
    def event_stream():
        started_at = time.time()
        chunks = []
        db = SessionLocal()
        try:
            for text_chunk in stream_generate_response(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
            ):
                chunks.append(text_chunk)
                yield f"data: {json.dumps({'token': text_chunk})}\n\n"

            full_response = "".join(chunks)
            log_inference(
                db=db,
                user_id=user.id,
                prompt=request.prompt,
                response=full_response,
                latency_ms=(time.time() - started_at) * 1000,
                success=True,
            )
            yield "data: [DONE]\n\n"
        except Exception as exc:
            log_inference(
                db=db,
                user_id=user.id,
                prompt=request.prompt,
                response=f"Error: {exc}",
                latency_ms=(time.time() - started_at) * 1000,
                success=False,
            )
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/logs", response_model=LogsResponse)
def get_logs(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0), db: Session = Depends(get_db), user: User = Depends(current_user)):
    """
    Retrieve recent inference logs.
    Args:
        limit: number of logs per page (default 50, maximum 100)
        offset: number of logs to skip for pagination
        db   : database session
    Returns:
        LogsResponse with list of logs and total count
    """
    query = db.query(InferenceLog).filter(InferenceLog.user_id == user.id)
    logs = query\
             .order_by(InferenceLog.timestamp.desc())\
             .offset(offset)\
             .limit(limit)\
             .all()
    total = query.count()

    return LogsResponse(logs=logs, total=total)
