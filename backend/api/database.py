"""
Database configuration for CodeMentor-LLM API
SQLite database for logging inference requests.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./codementor.db" # Local file will be make no server needed, just a file in the project directory

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} # handle multiple requests in SQLite
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) # Each request gets its own session to ensure thread safety and proper resource management

# Base class
Base = declarative_base() # All database models will inherit from this base class


# This creates a table
class InferenceLog(Base):
    """Model for logging inference requests."""
    __tablename__ = "inference_logs"

    id         = Column(Integer, primary_key=True, index=True)
    prompt     = Column(String, nullable=False)
    response   = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    success    = Column(Boolean, default=True)


# Create tables in the database
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

# Gives a database session for each API request
def get_db():
    """Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Function to log inference requests to the database
def log_inference(db, prompt: str, response: str, latency_ms: float, success: bool):
    """
    Log inference request to database.
    Args:
        db      : database session
        prompt  : user prompt
        response: model response
        latency_ms: inference latency
        success : whether inference was successful
    """
    log_entry = InferenceLog(
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        success=success
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry