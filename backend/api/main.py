"""
Main FastAPI Application for CodeMentor-LLM
Entry point for the REST API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from api.routes import router
from api.database import create_tables

if os.getenv("USE_LOCAL_MODEL", "false").lower() == "true":
    from api.model_loader_prod import load_model, generate_response
else:
    from api.model_loader import load_model, generate_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs startup and shutdown events.
    """
    # Startup
    print("Starting CodeMentor-LLM API...")
    create_tables()
    load_model()
    print("API ready")
    yield
    # Shutdown
    print("Shutting down CodeMentor-LLM API...")


# Create FastAPI app
app = FastAPI(
    title="CodeMentor-LLM API",
    description="REST API for CodeMentor-LLM — a fine-tuned coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )