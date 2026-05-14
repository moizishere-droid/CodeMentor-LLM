"""
Tests for FastAPI endpoints.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

from api.database import create_tables

@pytest.fixture(autouse=True)
def setup_db():
    create_tables()

client = TestClient(app)


def test_health_endpoint():
    """Test /health returns 200 and correct response."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "Abdulmoiz123/codementor-llm-merged"
    assert data["version"] == "1.0.0"


def test_generate_endpoint_valid():
    """Test /generate with valid prompt."""
    mock_result = {
        "response": "def reverse(s): return s[::-1]",
        "latency_ms": 100.0,
        "success": True
    }
    with patch("api.model_loader.generate_response", return_value=mock_result):
        response = client.post(
            "/generate",
            json={
                "prompt": "Write a Python function to reverse a string.",
                "max_new_tokens": 512
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "response" in data
    assert "latency_ms" in data


def test_generate_endpoint_empty_prompt():
    """Test /generate with empty prompt fails validation."""
    response = client.post(
        "/generate",
        json={
            "prompt": "",
            "max_new_tokens": 512
        }
    )
    assert response.status_code == 422


def test_generate_endpoint_missing_prompt():
    """Test /generate with missing prompt fails."""
    response = client.post(
        "/generate",
        json={"max_new_tokens": 512}
    )
    assert response.status_code == 422


def test_logs_endpoint():
    """Test /logs returns 200."""
    response = client.get("/logs")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert "total" in data
    assert isinstance(data["logs"], list)
    assert isinstance(data["total"], int)


def test_logs_endpoint_with_limit():
    """Test /logs with limit parameter."""
    response = client.get("/logs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert len(data["logs"]) <= 5

def test_generate_endpoint_max_tokens_limit():
    """Test /generate with max_new_tokens over limit fails."""
    response = client.post(
        "/generate",
        json={
            "prompt": "test",
            "max_new_tokens": 2000
        }
    )
    assert response.status_code == 422