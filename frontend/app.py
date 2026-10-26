"""CodeMentor-LLM's Streamlit workspace; API contracts are unchanged."""

from datetime import datetime
import base64
from pathlib import Path
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


LOGIN_INTRO = """
<style>.cm-login-intro { --cm-feature-reference: url(data:image/jpeg;base64,ARTWORK_DATA); }</style>
<div class="cm-login-intro">
  <span class="cm-landing-eyebrow">YOUR CODING WORKSPACE</span>
  <h1>Welcome back to<br>Code Mentor LLM</h1>
  <p class="cm-landing-description">Log in to continue your coding journey.<br>Save your progress, access your workspaces,<br>and pick up right where you left off.</p>
  <div class="cm-landing-features">
    <div><span class="cm-feature-icon">&lt;/&gt;</span><section><h3>Build and learn</h3><p>Work on real projects with AI support.</p></section></div>
    <div><span class="cm-feature-icon">▤</span><section><h3>Save your progress</h3><p>Keep your conversations and code safe.</p></section></div>
    <div><span class="cm-feature-icon">ϟ</span><section><h3>Grow your skills</h3><p>Get help, feedback, and new perspectives.</p></section></div>
  </div>
  <div class="cm-reference-art" role="img" aria-label="Coding workspace illustration" style="background-image:url(data:image/jpeg;base64,ILLUSTRATION_DATA)"></div>
</div>
"""


@st.dialog("Account notification", width="small")
def show_account_message(message: str, *, error: bool = True) -> None:
    """Keep notifications outside the fitted account page layout."""
    if error:
        st.error(message)
    else:
        st.info(message)
    if st.button("OK", key="dismiss_account_message", use_container_width=True):
        st.rerun()


def render_account() -> None:
    st.markdown('<span class="cm-auth-page" aria-hidden="true"></span>', unsafe_allow_html=True)
    mode = st.session_state.get("account_mode", "Log in")
    st.markdown('<span class="cm-login-landing" aria-hidden="true"></span>', unsafe_allow_html=True)
    account_key = "signup" if mode == "Sign up" else "login"
    alternative = "Sign up" if mode == "Log in" else "Log in"
    brand, action = st.columns([6, 2])
    with brand:
        st.markdown('<div class="cm-landing-brand"><span>&lt;/&gt;</span> Code Mentor LLM</div>', unsafe_allow_html=True)
    with action:
        st.markdown('<span class="cm-top-action"></span>', unsafe_allow_html=True)
        st.caption("New to Code Mentor LLM?" if mode == "Log in" else "Already have an account?")
        st.button(alternative, key="landing_signup", on_click=set_account_mode, args=(alternative,), use_container_width=True)
    introduction, account_column = st.columns([1.08, 1], gap="large")
    with introduction:
        artwork = base64.b64encode((Path(__file__).parent / "ui/assets/login-reference.jpg").read_bytes()).decode("ascii")
        intro = LOGIN_INTRO
        if mode == "Sign up":
            intro = intro.replace("Welcome back to<br>Code Mentor LLM", "Start your journey<br>with Code Mentor LLM").replace(
                "Log in to continue your coding journey.<br>Save your progress, access your workspaces,<br>and pick up right where you left off.",
                "Create an account to save your progress,<br>access your workspaces, and pick up right<br>where you left off.")
        illustration = artwork if mode == "Log in" else base64.b64encode((Path(__file__).parent / "ui/assets/signup-reference.jpg").read_bytes()).decode("ascii")
        st.markdown(intro.replace("ARTWORK_DATA", artwork).replace("ILLUSTRATION_DATA", illustration), unsafe_allow_html=True)
    with account_column:
        with st.container():
            st.markdown('<span class="cm-account-card" aria-hidden="true"></span>', unsafe_allow_html=True)
            title = "Welcome back to Code Mentor LLM" if mode == "Log in" else "Create your Code Mentor LLM account"
            description = "Log in to your account to save and revisit your responses." if mode == "Log in" else "Start your coding journey and keep your responses in one place."
            subtitle = f"<p>{description}</p>" if mode == "Log in" else ""
            st.markdown(f'<div class="cm-login-title"><h2>{title}</h2>{subtitle}</div>', unsafe_allow_html=True)
            with st.form(f"account_{account_key}", clear_on_submit=False):
                st.markdown('<span class="cm-auth-form"></span>', unsafe_allow_html=True)
                username = st.text_input("Username", key=f"{account_key}_username", autocomplete="username", max_chars=80, placeholder="Enter your username" if mode == "Log in" else "Choose a username")
                email = st.text_input("Email", key="signup_email", autocomplete="email", max_chars=254, placeholder="Enter your email address") if mode == "Sign up" else ""
                password = st.text_input("Password", key=f"{account_key}_password", autocomplete="new-password" if mode == "Sign up" else "current-password", type="password", max_chars=128, placeholder="Enter your password" if mode == "Log in" else "Create a password")
                confirm = st.text_input("Confirm password", key="signup_confirm_password", autocomplete="new-password", type="password", max_chars=128, placeholder="Confirm your password") if mode == "Sign up" else password
                if mode == "Log in":
                    remember_column, recovery_column = st.columns([1, 1])
                    with remember_column:
                        st.checkbox("Remember me", value=True, key="remember_account")
                    with recovery_column:
                        recovery = st.form_submit_button("Forgot password?")
                else:
                    recovery = False
                    st.markdown('<span class="cm-signup-fields"></span>', unsafe_allow_html=True)
                    agreed = st.checkbox("I agree to the Terms of Service and Privacy Policy.", key="account_agreed")
                submitted = st.form_submit_button(mode + "  →")
            st.markdown('<div class="cm-login-divider"><span>OR</span></div>', unsafe_allow_html=True)
            st.markdown('<span class="cm-account-footer"></span>', unsafe_allow_html=True)
            st.caption("Don’t have an account?" if mode == "Log in" else "Already have an account?")
            st.button(alternative + " →", key="account_switch", on_click=set_account_mode,
                      args=(alternative,), use_container_width=True)
    if recovery:
        show_account_message("Password recovery is not configured yet. Please contact the app owner for help accessing your account.", error=False)
    if submitted:
        if mode == "Sign up" and (not email.strip() or "@" not in email or "." not in email.rsplit("@", 1)[-1]):
            show_account_message("Enter a valid email address.")
        elif mode == "Sign up" and not agreed:
            show_account_message("Please agree to the Terms of Service and Privacy Policy to continue.")
        elif len(username.strip()) < 3 or len(password) < 8:
            show_account_message("Use a username of at least 3 characters and a password of at least 8 characters.")
        elif password != confirm:
            show_account_message("The passwords do not match.")
        else:
            try:
                endpoint = "register" if mode == "Sign up" else "login"
                payload = {"username": username, "password": password}
                if mode == "Sign up":
                    payload.update(email=email.strip(), terms_accepted=agreed)
                response = requests.post(f"{API_URL}/auth/{endpoint}", json=payload, timeout=30)
                if not response.ok:
                    detail = response.json().get("detail", "Unable to sign in")
                    show_account_message(detail if isinstance(detail, str) else "Check your username and password.")
                    return
                account = response.json()
                st.session_state.clear()
                st.session_state.auth_token = account["token"]
                st.session_state.username = account["username"]
                st.rerun()
            except (requests.RequestException, ValueError):
                show_account_message("The account service is unavailable. Please try again.")


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
    generation_indicator = st.empty()
    scenery = base64.b64encode((Path(__file__).parent / "ui/assets/workspace-reference.jpg").read_bytes()).decode("ascii")
    response_class = " cm-workspace-response" if has_asked else ""
    st.markdown(f'<style>.block-container:has(.cm-workspace-page){{--cm-workspace-reference:url(data:image/jpeg;base64,{scenery});}}</style>'
                f'<span class="cm-workspace-page{response_class}" aria-hidden="true"></span>'
                f'<div class="cm-workspace-art cm-workspace-art-left" style="background-image:url(data:image/jpeg;base64,{scenery})" aria-hidden="true"></div>'
                f'<div class="cm-workspace-art cm-workspace-art-right" style="background-image:url(data:image/jpeg;base64,{scenery})" aria-hidden="true"></div>', unsafe_allow_html=True)
    if not has_asked:
        with intro.container():
            st.markdown('<span class="cm-workspace-home" aria-hidden="true"></span>', unsafe_allow_html=True)
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
                height=100, key="prompt", label_visibility="collapsed",
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
                ("**Find the bug**\n\nPaste your code and get help finding issues.", "Help me debug this Python function."),
                ("**Understand a concept**\n\nGet clear explanations with examples.", "Explain Python list comprehensions with a simple example."),
                ("**Build something**\n\nTurn your idea into working code step by step.", "Write a Python function to reverse a string."),
            )
            for column, (label, suggestion) in zip(st.columns(3), suggestions):
                column.button(label, on_click=set_prompt, args=(suggestion,), use_container_width=True)
    else:
        suggestions_area.empty()

    if submitted and not question.strip():
        st.warning("Enter a coding question before sending.")
    elif submitted:
        generation_indicator.markdown('<span class="cm-generating" aria-hidden="true"></span>', unsafe_allow_html=True)
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
        finally:
            generation_indicator.empty()
        if "response_error" in st.session_state:
            feedback_area.error(st.session_state.response_error)

    if has_asked or submitted:
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


def start_new_request() -> None:
    for key in ("answer", "has_asked", "response_error", "active_request", "prompt", "draft"):
        st.session_state.pop(key, None)
    st.session_state.page = "workspace"


def choose_log_filter(period: str) -> None:
    st.session_state.log_period = period
    st.session_state.log_page = 1
    st.session_state.pop("active_request", None)


def change_log_page(delta: int) -> None:
    st.session_state.log_page = max(1, st.session_state.get("log_page", 1) + delta)


def reset_log_page() -> None:
    st.session_state.log_page = 1


def render_logs() -> None:
    st.markdown('<span class="cm-log-page" aria-hidden="true"></span>', unsafe_allow_html=True)
    filters, history = st.columns([1, 4.5], gap="large")
    with history:
        heading, search_column = st.columns([2.4, 1.5])
        with heading:
            render_heading("WORKSPACE ACTIVITY", "Request log", "Review your questions, responses, and request status.", wide=True)
        with search_column:
            search = st.text_input("Search requests", placeholder="Search your requests…", label_visibility="collapsed", key="log_search", on_change=reset_log_page)
        st.caption("Your saved responses · 50 per page")
    period = st.session_state.get("log_period", "all")
    page_number = st.session_state.get("log_page", 1)
    signature = (search, period, page_number)
    try:
        if signature != st.session_state.get("log_signature") or "request_log_data" not in st.session_state:
            response = requests.get(f"{API_URL}/logs", params={"limit": 50, "offset": (page_number - 1) * 50,
                                    "search": search, "period": period, "sort": "newest"},
                                    headers=auth_headers(), timeout=30)
            require_session(response)
            response.raise_for_status()
            st.session_state.request_log_data = response.json()
            st.session_state.log_signature = signature
        data = st.session_state.request_log_data
        counts = data.get("counts", {})
        with filters:
            with st.container(border=True):
                st.markdown('<span class="cm-log-filters" aria-hidden="true"></span>', unsafe_allow_html=True)
                st.button("＋  New Request", on_click=start_new_request, type="primary", use_container_width=True)
                for value, label in [("all", "▦  All Requests"), ("today", "▣  Today"), ("week", "▣  Last 7 days"), ("month", "▣  Last 30 days"), ("starred", "☆  Starred")]:
                    st.button(f"{label}    {counts.get(value, data['total'] if value == 'all' else 0)}", key=f"filter_{value}",
                              on_click=choose_log_filter, args=(value,), type="primary" if period == value else "secondary", use_container_width=True)
        with history:
            if not data["logs"]:
                st.info("No saved requests match this view. Start a new request or change your filters.")
            for record in data["logs"]:
                with st.container(border=True):
                    st.markdown('<span class="cm-log-row" aria-hidden="true"></span>', unsafe_allow_html=True)
                    icon, details, timestamp, actions = st.columns([.45, 5.5, 2, .45])
                    with icon:
                        st.markdown('<span class="cm-history-icon">☏</span>', unsafe_allow_html=True)
                    with details:
                        prompt = " ".join(str(record["prompt"]).split())
                        title = prompt[:70] + ("…" if len(prompt) > 70 else "")
                        if st.button(title, key=f"open_log_{record['id']}", use_container_width=True):
                            st.session_state.active_request = None if st.session_state.get("active_request") == record["id"] else record["id"]
                        st.caption(prompt[:120] + ("…" if len(prompt) > 120 else ""))
                    with timestamp:
                        raw = str(record.get("timestamp", ""))
                        try:
                            label = datetime.fromisoformat(raw).strftime("%b %d, %I:%M %p")
                        except ValueError:
                            label = ""
                        st.markdown(f'<div class="cm-log-meta"><time>{escape(label)}</time><span class="cm-message-count">2 messages</span></div>', unsafe_allow_html=True)
                    with actions:
                        with st.popover("⋮"):
                            if st.button("Unstar" if record.get("starred") else "Star request", key=f"star_{record['id']}"):
                                result = requests.patch(f"{API_URL}/logs/{record['id']}/star", json={"starred": not record.get("starred", False)}, headers=auth_headers(), timeout=30)
                                require_session(result)
                                result.raise_for_status()
                                st.session_state.pop("request_log_data", None)
                                st.rerun()
                    if st.session_state.get("active_request") == record["id"]:
                        render_log_record(record)
            total_pages = max(1, (data["total"] + 49) // 50)
            total, previous, current, following = st.columns([8, .6, .6, .6])
            total.caption(f"Total requests: {data['total']}")
            previous.button("←", disabled=page_number <= 1, on_click=change_log_page, args=(-1,), key="previous_logs")
            current.button(str(page_number), disabled=True, key="current_logs", type="primary")
            following.button("→", disabled=page_number >= total_pages, on_click=change_log_page, args=(1,), key="next_logs")
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        with history:
            st.error(f"We couldn’t load your requests: {error}")
            if st.button("Retry loading requests"):
                st.session_state.pop("request_log_data", None)
                st.rerun()


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
