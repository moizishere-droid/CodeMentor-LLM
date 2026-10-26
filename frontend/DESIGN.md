# CodeMentor-LLM interface

The existing Streamlit 1.35 application remains the frontend. No backend route,
request payload, model, or database changes are required.

## Screens

- Workspace: welcome copy and suggestions appear initially. Sending a question
  replaces this welcome section with the streamed answer and puts a cleared
  follow-up composer below it. A subsequent question replaces the last answer.
- Request Log: full-width cards show ID, prompt, complete Markdown response, and
  Success/Failed status. Responses are fully open, with no table cells, collapsed
  sections, or nested vertical scrolling. Latency and timestamps stay in the API
  but are omitted from the interface. Refresh, totals, empty and error states
  remain. This is the existing admin view, without new authorization behavior.
- Header: CodeMentor-LLM at left; Workspace, Request Log, and Appearance at right.
  Navigation uses soft active backgrounds and a 180ms hover transition. At small
  widths it becomes a row of accessible icons. All header targets are 44px.

## Design tokens

`ui/tokens.css` is the source for light/dark semantic colors, the green accent,
4/8/12/16/24/32/48px spacing, 12/20px and pill radii, subtle shadow, system sans
font stack, 12/16px body scale, responsive heading size, and 180ms transitions.
Inter is used if already installed; no remote font requests are necessary.

`ui/app.css` styles native Streamlit controls using their test identifiers.
`ui/index.html` is a small, dependency-free Streamlit component using its v1
message protocol. It controls only the selected view and light/dark appearance.
Because the static component is served from the app's own origin, theme changes
update the CSS appearance property in both documents in one browser task. The
server persists that preference; revision acknowledgements prevent older reruns
from reverting a newer selection. Color changes animate for 180ms and respect
reduced motion. Tooltips use explicit theme-aware foreground/background colors.
Inline code in answers has no background; fenced code remains formatted and
wraps long lines without a fixed vertical height.
Appearance, drafts, and the last completed response are session-local. Native
Streamlit developer chrome is hidden; application navigation is in the header.

The textarea retains Streamlit's native keyboard behavior and manual resizing.
Automatic height growth would require a richer composer component or upgrading
Streamlit.

The combined Hugging Face deployment contains matching `streamlit_app.py` and
`ui/` assets because its Docker context cannot import files from `frontend/`.
Keep these copies synchronized when changing the interface.

## Verification

Install `requirements-dev.txt` into a development environment, then run from the
repository root:

```powershell
python -m compileall -q frontend hf-combined/streamlit_app.py hf-combined/ui
python -m ruff check frontend
python -m mypy frontend/app.py frontend/ui --ignore-missing-imports --follow-imports=skip
python -m pytest frontend/tests -q
python -m streamlit run frontend/app.py
# In another terminal, with Chrome installed:
python frontend/tests/browser_check.py
python frontend/tests/browser_polish.py
```

Set `UI_URL` and `CHROME_PATH` if using another local port or Chrome installation.
Browser checks cover 360, 768, and 1440px in both appearances, navigation, overflow,
theme propagation, Appearance tooltip, keyboard focus ring, and touch target size.
Screenshots and browser results are saved in `.ui-checks/` (ignored by Git).
HTTP behavior is mocked in the frontend tests; live generation still requires the
existing FastAPI backend and model service.

Follow-ups: automatically growing composer, durable conversation history,
system-theme preference persistence, and a CI job to run these UI checks. These
are outside the current presentation refactor.
