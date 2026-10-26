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
RESPONSE_BUDGETS = {"Normal": 512, "Detailed code": 1024}
STREAM_RENDER_INTERVAL = 0.08

st.set_page_config(
    page_title="CodeMentor-LLM | Coding workspace",
    page_icon="💻",
    layout="wide",
    menu_items={"About": "CodeMentor-LLM — a coding assistant powered by Llama 3.2."},
)


def stream_response(prompt: str, max_new_tokens: int = 512) -> Iterator[str]:
    """Consume the existing server-sent token stream without altering its payload."""
    with requests.post(
        f"{API_URL}/generate/stream",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        stream=True,
        timeout=(30, 180),
    ) as response:
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


def continuation_prompt(question: str, answer: str) -> str:
    """Supply the original question and recent ending within the API's 2048 chars."""
    instructions = (
        "Continue the unfinished assistant answer below. Output only the missing continuation, "
        "including any needed leading whitespace or newlines. Do not repeat existing text. "
        "If the answer ends inside a code block, continue it without opening another fence.\n\n"
    )
    context = f"Original coding question:\n{question[:600]}\n\nAnswer ending:\n"
    ending_label = "\n\nContinue exactly after this ending:\n"
    remaining = 2048 - len(instructions) - len(context) - len(ending_label)
    return instructions + context + answer[-remaining:] + ending_label


def append_continuation(prefix: str, addition: str) -> str:
    """Remove only exact repeated boundary text; otherwise retain model output."""
    for length in range(min(len(prefix), len(addition), 512), 15, -1):
        if prefix.endswith(addition[:length]):
            return prefix + addition[length:]
    return prefix + addition


def render_response_stream(prompt: str, max_new_tokens: int, prefix: str = "") -> str:
    """Batch Markdown updates without the per-character typewriter cursor."""
    output = st.empty()
    if prefix:
        output.markdown(prefix)
    else:
        output.markdown('<p class="cm-waiting" role="status">Working on your question…</p>', unsafe_allow_html=True)
    chunks: list[str] = []
    last_render = 0.0
    for chunk in stream_response(prompt, max_new_tokens):
        chunks.append(chunk)
        st.session_state.answer = append_continuation(prefix, "".join(chunks))
        now = monotonic()
        if now - last_render >= STREAM_RENDER_INTERVAL:
            output.markdown(st.session_state.answer)
            last_render = now
    addition = "".join(chunks)
    if not addition.strip():
        raise RuntimeError("The service returned an empty answer. Please try again.")
    if prefix and addition.strip() == prefix.strip():
        st.session_state.answer = prefix
        raise RuntimeError("The model repeated the previous answer. Try a more specific follow-up question.")
    answer = append_continuation(prefix, addition)
    if prefix and answer == prefix:
        raise RuntimeError("No new content was returned. Try a more specific follow-up question.")
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
    continuation_area = st.empty()
    next_question = st.empty()
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
    continuation_rendered = bool(st.session_state.get("answer"))
    continue_requested = False
    if continuation_rendered:
        with continuation_area.container():
            continue_requested = st.button(
                "Continue response", key="continue_response",
                help="Ask for the remaining answer if it ended early. This sends another request.",
            )
    if has_asked:
        next_question.markdown(
            '<p class="cm-next-question">Ask another coding question</p>', unsafe_allow_html=True
        )

    selected_mode = st.radio(
        "Response length", list(RESPONSE_BUDGETS), horizontal=True,
        label_visibility="collapsed", key="response_mode",
        help="Normal uses up to 512 tokens. Detailed code allows up to 1024 tokens and may take longer.",
    )
    token_budget = RESPONSE_BUDGETS[selected_mode]

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
    elif submitted or continue_requested:
        prefix = str(st.session_state.get("answer", "")) if continue_requested else ""
        if continue_requested:
            question = continuation_prompt(str(st.session_state.get("last_question", "")), prefix)
        else:
            st.session_state.last_question = question
        intro.empty()
        suggestions_area.empty()
        response_area.empty()
        feedback_area.empty()
        next_question.markdown(
            '<p class="cm-next-question">Ask another coding question</p>', unsafe_allow_html=True
        )
        st.session_state.has_asked = True
        st.session_state.draft = ""
        if not continue_requested:
            st.session_state.pop("answer", None)
        st.session_state.pop("response_error", None)
        try:
            with response_area.container(), st.chat_message("assistant"):
                st.session_state.answer = render_response_stream(question, token_budget, prefix)
        except requests.exceptions.ConnectionError:
            st.session_state.response_error = "The coding service is unavailable. Please try again shortly."
        except requests.exceptions.Timeout:
            st.session_state.response_error = "The request took too long. Please try again."
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as error:
            st.session_state.response_error = f"We couldn’t complete your request: {error}"
        if "response_error" in st.session_state:
            feedback_area.error(st.session_state.response_error)
        if not continuation_rendered and st.session_state.get("answer"):
            with continuation_area.container():
                st.button(
                    "Continue response", key="continue_response",
                    help="Ask for the remaining answer if it ended early. This sends another request.",
                )

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
    st.caption("Admin view · Latest 50 requests")
    refresh = st.button("Refresh logs")
    try:
        if refresh or "request_log_data" not in st.session_state:
            with st.spinner("Loading requests…"):
                response = requests.get(f"{API_URL}/logs?limit=50", timeout=30)
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


if render_shell() == "logs":
    render_logs()
else:
    render_workspace()
