"""
Database configuration for CodeMentor-LLM API
SQLite database for logging inference requests.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database URL
DB_PATH = Path(os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parents[1] / "codementor.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}" # Local file will be make no server needed, just a file in the project directory

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
    user_id    = Column(Integer, index=True, nullable=True)
    prompt     = Column(String, nullable=False)
    response   = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    success    = Column(Boolean, default=True)
    starred    = Column(Boolean, default=False, nullable=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String(120), nullable=True)
    email = Column(String(254), nullable=True)
    terms_accepted_at = Column(DateTime, nullable=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash = Column(String(64), primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)


# Create tables in the database
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    user_columns = {column["name"] for column in inspect(engine).get_columns("users")}
    with engine.begin() as connection:
        for name, sql_type in [("full_name", "VARCHAR(120)"), ("email", "VARCHAR(254)"), ("terms_accepted_at", "DATETIME")]:
            if name not in user_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {name} {sql_type}"))
    if "starred" not in {column["name"] for column in inspect(engine).get_columns("inference_logs")}:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE inference_logs ADD COLUMN starred BOOLEAN NOT NULL DEFAULT 0"))
    if "user_id" not in {column["name"] for column in inspect(engine).get_columns("inference_logs")}:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE inference_logs ADD COLUMN user_id INTEGER"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_inference_logs_user_id ON inference_logs (user_id)"))

# Gives a database session for each API request
def get_db():
    """Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Function to log inference requests to the database
def log_inference(db, prompt: str, response: str, latency_ms: float, success: bool, user_id: int | None = None):
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
        user_id=user_id,
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        success=success
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry