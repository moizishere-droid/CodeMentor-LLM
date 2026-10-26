"""CodeMentor-LLM's Streamlit workspace; API contracts are unchanged."""

import json
import os
from collections.abc import Iterator

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from ui import render_heading, render_shell

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CodeMentor-LLM | Coding workspace",
    page_icon="💻",
    layout="wide",
    menu_items={"About": "CodeMentor-LLM — a coding assistant powered by Llama 3.2."},
)


def stream_response(prompt: str) -> Iterator[str]:
    """Consume the existing server-sent token stream without altering its payload."""
    with requests.post(
        f"{API_URL}/generate/stream",
        json={"prompt": prompt, "max_new_tokens": 512},
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


def set_prompt(value: str) -> None:
    st.session_state.prompt = value


def render_workspace() -> None:
    st.session_state.setdefault("prompt", st.session_state.get("draft", ""))
    render_heading(
        "A CLEARER WAY TO CODE",
        "Let’s work it out, together.",
        "Untangle a bug, explore an idea, or build something new.",
    )
    with st.form("question", clear_on_submit=False):
        prompt = st.text_area(
            "Your coding question",
            placeholder="What are you working on?",
            height=120,
            key="prompt",
            label_visibility="collapsed",
        )
        model, send = st.columns([6, 1])
        with model:
            st.markdown(
                '<span class="cm-model">Llama 3.2 · Coding assistant</span>',
                unsafe_allow_html=True,
            )
        with send:
            submitted = st.form_submit_button("Send question", type="primary", help="Send question")

    suggestions = (
        ("Find the bug", "Help me debug this Python function."),
        ("Understand a concept", "Explain Python list comprehensions with a simple example."),
        ("Build something", "Write a Python function to reverse a string."),
    )
    for column, (label, suggestion) in zip(st.columns(3), suggestions):
        column.button(label, on_click=set_prompt, args=(suggestion,), use_container_width=True)

    if submitted and not prompt.strip():
        st.warning("Enter a coding question before sending.")
    elif submitted:
        st.session_state.pop("answer", None)
        try:
            with st.chat_message("assistant"), st.spinner("Working on your question…"):
                st.session_state.answer = st.write_stream(stream_response(prompt))
        except requests.exceptions.ConnectionError:
            st.error("The coding service is unavailable. Please try again shortly.")
        except requests.exceptions.Timeout:
            st.error("The request took too long. Please try again.")
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as error:
            st.error(f"We couldn’t complete your request: {error}")
    elif "answer" in st.session_state:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.answer)

    st.markdown(
        '<p class="cm-footer">AI can make mistakes. Review and test generated code.</p>',
        unsafe_allow_html=True,
    )


def render_logs() -> None:
    render_heading(
        "WORKSPACE ACTIVITY", "Request log", "Review recent requests and their generation details."
    )
    st.caption("Admin view · Latest 50 requests")
    st.button("Refresh logs")
    try:
        with st.spinner("Loading requests…"):
            response = requests.get(f"{API_URL}/logs?limit=50", timeout=30)
            response.raise_for_status()
            data = response.json()
        st.caption(f"Total requests: {data['total']}")
        if data["logs"]:
            frame = pd.DataFrame(data["logs"])
            frame = frame[["id", "prompt", "response", "latency_ms", "timestamp", "success"]]
            st.dataframe(frame, use_container_width=True, hide_index=True)
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
