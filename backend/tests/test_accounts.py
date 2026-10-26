"""Account boundaries and persisted history, without calling the model."""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from api import database, routes
from api.database import Base, AuthSession, get_db
from fastapi import FastAPI
from api.auth import router as auth_router
app = FastAPI()
app.include_router(routes.router)
app.include_router(auth_router)

@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'accounts.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    def database_session():
        with factory() as session:
            yield session
    app.dependency_overrides[get_db] = database_session
    monkeypatch.setattr(routes, "SessionLocal", factory)
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, factory
    app.dependency_overrides.clear()
    engine.dispose()


def register(client, name):
    response = client.post("/auth/register", json={"username": name, "password": "password123"})
    assert response.status_code == 201
    return {"Authorization": "Bearer " + response.json()["token"]}


def test_accounts_history_and_logout(client):
    client, factory = client
    alice = register(client, "Alice")
    bob = register(client, "Bob")
    assert client.get("/logs").status_code == 401
    assert client.post("/generate", json={"prompt": "hello"}).status_code == 401
    assert client.post("/auth/register", json={"username": "ALICE", "password": "password123"}).status_code == 409
    assert client.post("/auth/login", json={"username": "alice", "password": "wrongpassword"}).status_code == 401
    with patch.object(routes, "generate_response", return_value={"response": "saved answer", "latency_ms": 1, "success": True}):
        assert client.post("/generate", headers=alice, json={"prompt": "hello"}).status_code == 200
    with patch.object(routes, "stream_generate_response", return_value=iter(["stream", " answer"])):
        response = client.post("/generate/stream", headers=alice, json={"prompt": "stream question"})
        assert "[DONE]" in response.text
    assert client.get("/logs", headers=bob).json()["total"] == 0
    assert client.get("/logs", headers=alice).json()["total"] == 2
    assert len(client.get("/logs?limit=1&offset=1", headers=alice).json()["logs"]) == 1
    assert client.post("/auth/logout", headers=alice).status_code == 200
    assert client.get("/logs", headers=alice).status_code == 401
    login = client.post("/auth/login", json={"username": "ALICE", "password": "password123"})
    restored = {"Authorization": "Bearer " + login.json()["token"]}
    assert client.get("/logs", headers=restored).json()["total"] == 2
    with factory() as session:
        for token in session.query(AuthSession).all():
            token.expires_at = datetime.utcnow() - timedelta(seconds=1)
        session.commit()
    assert client.get("/logs", headers=restored).status_code == 401


def test_existing_database_migration(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE inference_logs (id INTEGER PRIMARY KEY, prompt VARCHAR NOT NULL, response VARCHAR NOT NULL, latency_ms FLOAT NOT NULL, timestamp DATETIME, success BOOLEAN)"))
        connection.execute(text("INSERT INTO inference_logs (prompt,response,latency_ms,success) VALUES ('old prompt','old answer',1,1)"))
    monkeypatch.setattr(database, "engine", engine)
    database.create_tables()
    database.create_tables()
    assert "user_id" in {c["name"] for c in inspect(engine).get_columns("inference_logs")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT response, user_id FROM inference_logs")).one() == ("old answer", None)
    engine.dispose()
