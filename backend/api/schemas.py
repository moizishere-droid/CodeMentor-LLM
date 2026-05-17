"""
Pydantic schemas for CodeMentor-LLM API
Defines request and response models.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class GenerateRequest(BaseModel):
    prompt: str = Field(...,description="Coding question or prompt",min_length=1,max_length=2048)
    max_new_tokens: int = Field(default=512,description="Maximum tokens to generate",ge=1,le=1024)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Write a Python function to reverse a string.",
                "max_new_tokens": 512
            }
        }


class GenerateResponse(BaseModel):
    response: str
    latency_ms: float
    success: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class LogEntry(BaseModel):
    id: int
    prompt: str
    response: str
    latency_ms: float
    timestamp: datetime
    success: bool

    class Config:
        from_attributes = True


class LogsResponse(BaseModel):
    logs: list[LogEntry]
    total: int