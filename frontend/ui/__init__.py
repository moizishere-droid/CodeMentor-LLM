"""Presentation helpers shared by the workspace and request log views."""

from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).parent
_header = components.declare_component("codementor_header", path=str(ASSETS))


def render_shell() -> str:
    """Render navigation and apply the selected appearance."""
    st.session_state.setdefault("page", "workspace")
    st.session_state.setdefault("appearance", "light")
    if "prompt" in st.session_state:
        st.session_state.draft = st.session_state.prompt
    # Widget state is available before rendering. Consume it first so the header
    # and page can finish in one run, rather than interrupting their own render.
    event = st.session_state.get("main_navigation")
    if isinstance(event, dict) and event != st.session_state.get("last_header_event"):
        st.session_state.last_header_event = event
        if event.get("page") in ("workspace", "logs"):
            if event["page"] == "logs" and st.session_state.page != "logs":
                st.session_state.pop("request_log_data", None)
            st.session_state.page = event["page"]
        if event.get("appearance") in ("light", "dark"):
            st.session_state.appearance = event["appearance"]
    _header(
        page=st.session_state.page,
        appearance=st.session_state.appearance,
        ack_revision=st.session_state.get("last_header_event", {}).get("revision", 0),
        key="main_navigation",
        default=None,
    )

    tokens = (ASSETS / "tokens.css").read_text(encoding="utf-8")
    styles = (ASSETS / "app.css").read_text(encoding="utf-8")
    st.markdown(
        f"<style>{tokens}\n:root {{color-scheme: {st.session_state.appearance};"
        f" --cm-appearance: {st.session_state.appearance};}}"
        f"\n{styles}</style>",
        unsafe_allow_html=True,
    )
    return str(st.session_state.page)


def render_heading(eyebrow: str, title: str, description: str, *, wide: bool = False) -> None:
    """Render escaped page copy."""
    st.markdown(
        f'<div class="cm-heading{" cm-heading-wide" if wide else ""}">'
        f'<span class="cm-eyebrow">{escape(eyebrow)}</span>'
        f"<h1>{escape(title)}</h1><p>{escape(description)}</p></div>",
        unsafe_allow_html=True,
    )
