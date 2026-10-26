"""UI regression checks using Streamlit's native app runner and mocked HTTP."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))


def workspace() -> AppTest:
    return AppTest.from_file(str(FRONTEND / "app.py")).run()


def submit(app: AppTest, prompt: str) -> AppTest:
    app.text_area[0].input(prompt)
    return app.button[0].click().run()


def test_initial_workspace_and_empty_validation():
    app = workspace()
    assert not app.exception
    with patch("requests.post") as post:
        submit(app, "   ")
    post.assert_not_called()
    assert "Enter a coding question" in app.warning[0].value


def test_suggestion_fills_prompt_without_sending():
    app = workspace()
    with patch("requests.post") as post:
        app.button[3].click().run()
    post.assert_not_called()
    assert app.text_area[0].value == "Write a Python function to reverse a string."


def test_streaming_contract_and_response_survives_rerender():
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = [
        ": keep-alive", "", 'data: {"token": "Hello "}',
        'data: {"token": "world"}', "data: [DONE]", 'data: {"token": "ignored"}',
    ]
    app = workspace()
    with patch("requests.post", return_value=response) as post:
        submit(app, "Explain decorators")
    assert not app.exception
    assert app.session_state.answer == "Hello world"
    assert post.call_args.kwargs == {
        "json": {"prompt": "Explain decorators", "max_new_tokens": 512},
        "stream": True, "timeout": (30, 180),
    }
    app.session_state.appearance = "dark"
    app.run()
    assert app.session_state.answer == "Hello world"
    assert app.text_area[0].value == ""


@pytest.mark.parametrize("error", [requests.ConnectionError(), requests.Timeout(), RuntimeError("Failed")])
def test_generation_errors_are_visible(error):
    app = workspace()
    with patch("requests.post", side_effect=error):
        submit(app, "Explain decorators")
    assert not app.exception
    assert len(app.error) == 1


def test_logs_empty_and_http_error_states():
    app = workspace()
    app.session_state.page = "logs"
    response = MagicMock()
    response.json.return_value = {"total": 0, "logs": []}
    with patch("requests.get", return_value=response) as get:
        app.run()
    assert not app.exception
    assert "No requests yet" in app.info[0].value
    assert get.call_args.args[0].endswith("/logs?limit=50")
    response.raise_for_status.side_effect = requests.HTTPError("503")
    with patch("requests.get", return_value=response):
        app.run()
    assert "couldn’t load" in app.error[0].value


def test_logs_present_only_requested_fields_with_full_response():
    app = workspace()
    app.session_state.page = "logs"
    response = MagicMock()
    response.json.return_value = {"total": 1, "logs": [{
        "id": 1, "prompt": "Question", "response": "Answer", "latency_ms": 42,
        "timestamp": "2026-10-26T12:00:00", "success": True,
    }]}
    with patch("requests.get", return_value=response):
        app.run()
    assert not app.exception
    content = "\n".join(item.value for item in app.markdown)
    assert "Request #1" in content
    assert "Question" in content
    assert "Success" in content
    assert "Answer" in content
    assert "latency_ms" not in content
    assert "2026-10-26T12:00:00" not in content
    assert not app.expander
    assert not app.dataframe


def test_draft_survives_navigation():
    app = workspace()
    app.text_area[0].input("A question I am still writing").run()
    app.session_state.page = "logs"
    with patch("requests.get", side_effect=requests.ConnectionError()):
        app.run()
    app.session_state.page = "workspace"
    app.run()
    assert app.text_area[0].value == "A question I am still writing"


def test_combined_deployment_matches_frontend():
    combined = FRONTEND.parent / "hf-combined"
    assert (FRONTEND / "app.py").read_bytes() == (combined / "streamlit_app.py").read_bytes()
    for name in ("__init__.py", "tokens.css", "app.css", "index.html"):
        assert (FRONTEND / "ui" / name).read_bytes() == (combined / "ui" / name).read_bytes()


def test_response_replaces_welcome_and_followup_replaces_previous_answer():
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = ['data: {"token": "First answer"}', "data: [DONE]"]
    app = workspace()
    with patch("requests.post", return_value=response):
        submit(app, "First question")
    content = "\n".join(item.value for item in app.markdown)
    assert "Let’s work it out" not in content
    assert "First answer" in content
    assert "Ask another coding question" in content
    assert len(app.button) == 1
    response.iter_lines.return_value = ['data: {"token": "Second answer"}', "data: [DONE]"]
    with patch("requests.post", return_value=response):
        submit(app, "Second question")
    content = "\n".join(item.value for item in app.markdown)
    assert "Second answer" in content
    assert "First answer" not in content
    assert len(app.chat_message) == 1


def test_failed_log_status_and_escaped_prompt():
    app = workspace()
    app.session_state.page = "logs"
    response = MagicMock()
    response.json.return_value = {"total": 1, "logs": [{
        "id": 9, "prompt": "<script>alert('test')</script>", "response": "Service unavailable",
        "success": False, "latency_ms": 99, "timestamp": "hidden timestamp",
    }]}
    with patch("requests.get", return_value=response):
        app.run()
    content = "\n".join(item.value for item in app.markdown)
    assert "Status: Failed" in content
    assert "&lt;script&gt;" in content
    assert "<script>" not in content
    assert "hidden timestamp" not in content
