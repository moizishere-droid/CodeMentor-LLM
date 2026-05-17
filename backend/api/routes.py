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

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse, LogsResponse
from api.database import get_db, log_inference, InferenceLog
from api.model_loader import generate_response

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
def generate(request: GenerateRequest, db: Session = Depends(get_db)):
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
def get_logs(limit: int = 5, db: Session = Depends(get_db)):
    """
    Retrieve recent inference logs.
    Args:
        limit: number of logs to retrieve (default 5)
        db   : database session
    Returns:
        LogsResponse with list of logs and total count
    """
    logs = db.query(InferenceLog)\
             .order_by(InferenceLog.timestamp.desc())\
             .limit(limit)\
             .all()
    total = db.query(InferenceLog).count()

    return LogsResponse(logs=logs, total=total)