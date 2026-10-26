"""Check Modal endpoint budget normalization without a GPU or network calls."""

import asyncio
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def endpoint(monkeypatch):
    def decorator(**_kwargs):
        return lambda target: target

    modal = ModuleType("modal")
    modal.App = MagicMock(return_value=SimpleNamespace(cls=decorator, function=decorator))
    modal.Image = MagicMock()
    modal.Secret = MagicMock()
    modal.enter = decorator
    modal.method = decorator
    modal.fastapi_endpoint = decorator
    monkeypatch.setitem(sys.modules, "modal", modal)
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src" / "modal_app.py"))
    endpoint = module["generate_endpoint"]
    generate = MagicMock(return_value={"response": "Complete answer", "success": True})
    endpoint.__globals__["model"] = SimpleNamespace(generate=SimpleNamespace(remote=generate))
    return endpoint, generate


@pytest.mark.parametrize("requested,expected", [(512, 512), (1024, 1024), (9999, 1024), (0, 1), ("1024", 1024), (None, 512)])
def test_endpoint_respects_existing_api_budget(endpoint, requested, expected):
    generate_endpoint, generate = endpoint
    result = generate_endpoint({"prompt": "Write a linked list", "max_new_tokens": requested})
    generate.assert_called_once_with("Write a linked list", expected)
    assert result["success"] is True


def test_streaming_endpoint_passes_larger_budget_to_model(endpoint):
    generate_endpoint, _generate = endpoint
    stream = MagicMock(return_value=iter(["Complete output"]))
    generate_endpoint.__globals__["model"].generate_stream = SimpleNamespace(remote_gen=stream)
    response = generate_endpoint({"prompt": "Write a linked list", "max_new_tokens": 1024, "stream": True})

    async def collect():
        return [chunk async for chunk in response.body_iterator]

    events = asyncio.run(collect())
    stream.assert_called_once_with("Write a linked list", 1024)
    assert '"token": "Complete output"' in events[0]
    assert events[-1] == "data: [DONE]\n\n"
