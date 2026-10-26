"""CodeMentor-LLM's Streamlit workspace; API contracts are unchanged."""

import json
import os
from collections.abc import Iterator
from html import escape
from time import monotonic

import requests
import streamlit as st
from dotenv import load_dotenv
from ui import render_heading, render_shell

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_RESPONSE_TOKENS = 512
STREAM_RENDER_INTERVAL = 0.08

st.set_page_config(
    page_title="CodeMentor-LLM | Coding workspace",
    page_icon="💻",
    layout="wide",
    menu_items={"About": "CodeMentor-LLM — a coding assistant powered by Llama 3.2."},
)


def require_session(response: requests.Response) -> None:
    if response.status_code == 401:
        st.session_state.clear()
        st.rerun()


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}


def set_account_mode(mode: str) -> None:
    st.session_state.account_mode = mode


def render_account() -> None:
    st.markdown('<span class="cm-auth-page" aria-hidden="true"></span>', unsafe_allow_html=True)
    render_heading("YOUR CODING WORKSPACE", "Welcome to CodeMentor", "Log in or create an account to save and revisit your responses.")
    mode = st.session_state.get("account_mode", "Log in")
    with st.form("account"):
        st.markdown('<span class="cm-auth-form"></span>', unsafe_allow_html=True)
        username = st.text_input("Username", max_chars=80)
        password = st.text_input("Password", type="password", max_chars=128)
        confirm = st.text_input("Confirm password", type="password", max_chars=128) if mode == "Sign up" else password
        submitted = st.form_submit_button(mode)
    with st.container():
        st.markdown('<span class="cm-auth-switch" aria-hidden="true"></span>', unsafe_allow_html=True)
        login, signup = st.columns(2)
        login.button("Log in", key="account_login", on_click=set_account_mode,
                     args=("Log in",), type="primary" if mode == "Log in" else "secondary",
                     use_container_width=True)
        signup.button("Sign up", key="account_signup", on_click=set_account_mode,
                      args=("Sign up",), type="primary" if mode == "Sign up" else "secondary",
                      use_container_width=True)
    if submitted:
        if len(username.strip()) < 3 or len(password) < 8:
            st.error("Use a username of at least 3 characters and a password of at least 8 characters.")
        elif password != confirm:
            st.error("The passwords do not match.")
        else:
            try:
                endpoint = "register" if mode == "Sign up" else "login"
                response = requests.post(f"{API_URL}/auth/{endpoint}", json={"username": username, "password": password}, timeout=30)
                if not response.ok:
                    detail = response.json().get("detail", "Unable to sign in")
                    st.error(detail if isinstance(detail, str) else "Check your username and password.")
                    return
                account = response.json()
                st.session_state.clear()
                st.session_state.auth_token = account["token"]
                st.session_state.username = account["username"]
                # Restore the last saved answer immediately on login.
                history = requests.get(f"{API_URL}/logs?limit=1", headers=auth_headers(), timeout=30)
                if history.ok and history.json()["logs"]:
                    st.session_state.answer = history.json()["logs"][0]["response"]
                st.rerun()
            except (requests.RequestException, ValueError):
                st.error("The account service is unavailable. Please try again.")


def stream_response(prompt: str) -> Iterator[str]:
    """Consume the existing server-sent token stream without altering its payload."""
    with requests.post(
        f"{API_URL}/generate/stream",
        json={"prompt": prompt, "max_new_tokens": MAX_RESPONSE_TOKENS},
        headers=auth_headers(),
        stream=True,
        timeout=(30, 180),
    ) as response:
        require_session(response)
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if "token" in event:
                yield event["token"]
        raise RuntimeError("The response connection ended early. Please try your question again.")


def render_response_stream(prompt: str) -> str:
    """Batch Markdown updates without the per-character typewriter cursor."""
    output = st.empty()
    output.markdown('<p class="cm-waiting" role="status">Working on your question…</p>', unsafe_allow_html=True)
    chunks: list[str] = []
    last_render = 0.0
    for chunk in stream_response(prompt):
        chunks.append(chunk)
        st.session_state.answer = "".join(chunks)
        now = monotonic()
        if now - last_render >= STREAM_RENDER_INTERVAL:
            output.markdown(st.session_state.answer)
            last_render = now
    answer = "".join(chunks)
    if not answer.strip():
        raise RuntimeError("The service returned an empty answer. Please try again.")
    output.markdown(answer)
    return answer


def set_prompt(value: str) -> None:
    st.session_state.prompt = value


def prepare_question() -> None:
    """Capture the submitted text and clear only a valid, submitted draft."""
    if st.session_state.prompt.strip():
        st.session_state.submitted_question = st.session_state.prompt
        st.session_state.prompt = ""
        st.session_state.draft = ""
        st.session_state.has_asked = True


def render_workspace() -> None:
    st.session_state.setdefault("prompt", st.session_state.get("draft", ""))
    has_asked = st.session_state.get("has_asked", False) or "answer" in st.session_state
    intro = st.empty()
    response_area = st.empty()
    feedback_area = st.empty()
    if not has_asked:
        with intro.container():
            render_heading(
                "A CLEARER WAY TO CODE",
                "Let’s work it out, together.",
                "Untangle a bug, explore an idea, or build something new.",
            )
    if "answer" in st.session_state:
        with response_area.container(), st.chat_message("assistant"):
            st.markdown(st.session_state.answer)
    if "response_error" in st.session_state:
        feedback_area.error(st.session_state.response_error)
    with st.form("question", clear_on_submit=False):
        if has_asked:
            st.markdown('<span class="cm-compact-composer" aria-hidden="true"></span>', unsafe_allow_html=True)
            entry, send = st.columns([12, 1])
            with entry:
                prompt = st.text_area(
                    "Your coding question", placeholder="Ask another coding question…",
                    height=68, key="prompt", label_visibility="collapsed",
                )
        else:
            prompt = st.text_area(
                "Your coding question", placeholder="What are you working on?",
                height=120, key="prompt", label_visibility="collapsed",
            )
            model, send = st.columns([6, 1])
            with model:
                st.markdown(
                    '<span class="cm-model">Llama 3.2 · Coding assistant</span>',
                    unsafe_allow_html=True,
                )
        with send:
            submitted = st.form_submit_button(
                "Send question", type="primary", help="Send question", on_click=prepare_question
            )

    question = st.session_state.pop("submitted_question", prompt) if submitted else ""

    suggestions_area = st.empty()
    if not has_asked:
        with suggestions_area.container():
            suggestions = (
                ("Find the bug", "Help me debug this Python function."),
                ("Understand a concept", "Explain Python list comprehensions with a simple example."),
                ("Build something", "Write a Python function to reverse a string."),
            )
            for column, (label, suggestion) in zip(st.columns(3), suggestions):
                column.button(label, on_click=set_prompt, args=(suggestion,), use_container_width=True)

    if submitted and not question.strip():
        st.warning("Enter a coding question before sending.")
    elif submitted:
        intro.empty()
        suggestions_area.empty()
        response_area.empty()
        feedback_area.empty()
        st.session_state.has_asked = True
        st.session_state.draft = ""
        st.session_state.pop("answer", None)
        st.session_state.pop("response_error", None)
        try:
            with response_area.container(), st.chat_message("assistant"):
                st.session_state.answer = render_response_stream(question)
        except requests.exceptions.ConnectionError:
            st.session_state.response_error = "The coding service is unavailable. Please try again shortly."
        except requests.exceptions.Timeout:
            st.session_state.response_error = "The request took too long. Please try again."
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as error:
            st.session_state.response_error = f"We couldn’t complete your request: {error}"
        if "response_error" in st.session_state:
            feedback_area.error(st.session_state.response_error)

    st.markdown(
        '<p class="cm-footer">AI can make mistakes. Review and test generated code.</p>',
        unsafe_allow_html=True,
    )


def render_log_record(record: dict[str, object]) -> None:
    """Show each prompt and its complete answer without a nested scroll area."""
    succeeded = record["success"] is True
    status = "Success" if succeeded else "Failed"
    status_class = "cm-status-success" if succeeded else "cm-status-failed"
    with st.container(border=True):
        st.markdown(
            '<div class="cm-log-header">'
            f'<strong>Request #{escape(str(record["id"]))}</strong>'
            f'<span class="cm-status {status_class}" aria-label="Status: {status}">{status}</span>'
            '</div>'
            '<p class="cm-log-label">Prompt</p>'
            f'<p class="cm-log-prompt">{escape(str(record["prompt"]))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="cm-log-label">Response</p>', unsafe_allow_html=True)
        st.markdown(str(record["response"]))


def render_logs() -> None:
    render_heading(
        "WORKSPACE ACTIVITY", "Request log", "Review your questions, responses, and request status.",
        wide=True,
    )
    st.caption("Your saved responses · 50 per page")
    page_number = st.number_input("History page", min_value=1, value=1, step=1)
    refresh = st.button("Refresh logs")
    try:
        if refresh or "request_log_data" not in st.session_state or st.session_state.get("history_page") != page_number:
            with st.spinner("Loading requests…"):
                response = requests.get(f"{API_URL}/logs", params={"limit": 50, "offset": (page_number - 1) * 50}, headers=auth_headers(), timeout=30)
                st.session_state.history_page = page_number
                require_session(response)
                response.raise_for_status()
                st.session_state.request_log_data = response.json()
        data = st.session_state.request_log_data
        st.caption(f"Total requests: {data['total']}")
        if data["logs"]:
            records = {record["id"]: record for record in data["logs"]}
            request_ids = list(records)
            previous = st.session_state.get("active_request", request_ids[0])
            if st.session_state.get("request_choice") not in request_ids:
                st.session_state.request_choice = previous if previous in request_ids else request_ids[0]
            with st.sidebar:
                st.markdown("### Request history")
                selected = st.radio(
                    "Choose a request", request_ids,
                    format_func=lambda request_id: (
                        f"Request {request_id} · "
                        f"{' '.join(str(records[request_id]['prompt']).split())[:48]}"
                    ),
                    key="request_choice", label_visibility="collapsed",
                )
            st.session_state.active_request = selected
            render_log_record(records[selected])
        else:
            st.info("No requests yet. Send a coding question to see its details here.")
    except requests.exceptions.Timeout:
        st.error("Loading requests took too long. Select Refresh logs to try again.")
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        st.error(f"We couldn’t load the request log: {error}")


page = render_shell(show_navigation="auth_token" in st.session_state)
if "auth_token" not in st.session_state:
    render_account()
else:
    if page == "logout":
        try:
            response = requests.post(f"{API_URL}/auth/logout", headers=auth_headers(), timeout=30)
            if response.status_code != 401:
                response.raise_for_status()
        except requests.RequestException:
            st.session_state.page = "workspace"
            st.error("Could not log out. Please select Log out again to retry.")
            st.stop()
        st.session_state.clear()
        st.rerun()
    if page == "logs":
        render_logs()
    else:
        render_workspace()
