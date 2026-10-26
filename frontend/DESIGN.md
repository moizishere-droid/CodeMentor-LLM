# CodeMentor-LLM interface

The existing Streamlit 1.35 application remains the frontend. Backend routes,
request fields, model weights, and database schemas are unchanged. The frontend
uses 512 response tokens in Normal mode and up to 1024 when the user selects
Detailed code. Modal's serving
cap is aligned to that same maximum rather than silently reducing it to 512.
Deploying this cap correction requires redeploying `backend/src/modal_app.py` to
Modal as well as updating the frontend. A Space-only push cannot update Modal.
1024 remains a maximum, so arbitrarily long answers can still reach the limit.

## Screens

- Workspace: welcome copy and suggestions appear initially. Sending a question
  replaces this welcome section with the streamed answer and puts a cleared
  follow-up composer below it. A subsequent question replaces the last answer.
  The follow-up composer is a wider, compact dark bar in both themes. Native
  response headings have explicit theme-aware foregrounds and a stronger neutral
  card background; text-selection colors stay readable in both appearances.
- Request Log: the left sidebar lists Request 1, Request 2, and short prompt
  titles. Choosing an item opens its full prompt, Markdown response, and status
  in the main area. On mobile the history becomes a collapsible drawer.
  Responses are fully open, with no table cells, collapsed
  sections, or nested vertical scrolling. Latency and timestamps stay in the API
  but are omitted from the interface. Refresh, totals, empty and error states
  remain. History is fetched when opening the log view or refreshing; selecting
  an entry does not issue another API call. This is the existing admin view,
  without new authorization behavior.
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
The header consumes widget state before rendering, without an extra rerun.
Same-page clicks do nothing; rapid navigation changes are coalesced with one
request in flight and the latest selection queued. This keeps the iframe alive.
Inline code in answers has no background; fenced code remains formatted and
wraps long lines without a fixed vertical height.
Streaming Markdown updates are batched at 80ms rather than animated character
by character. Stable scrollbar space, disabled scroll anchoring, and a reserved
response area avoid horizontal jumps and automatic viewport movement. The
loading message is replaced in place rather than inserting/removing a spinner
above the answer. Interrupted streams retain the generated text and show an
explicit error; they are not reported as successfully completed responses.
Caret colors are explicit for both the light input and dark follow-up composer.

Normal mode is the default. Detailed code raises only the maximum output length
and can take longer. Continue response is a manual extra request, never an
automatic loop. It appends to the current answer using the original question
and a recent ending that fits the existing 2048-character prompt limit. Exact
repeated boundary text is trimmed; a wholly repeated answer shows an error.
The API does not report finish reasons, so the control is offered after any
answer instead of claiming to reliably detect token exhaustion. Continuations
have limited recent context and are not guaranteed to reproduce missing code
correctly; generated code should still be reviewed and tested.
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
python -m pytest backend/tests/test_generation_budget.py -q
python -m streamlit run frontend/app.py
# In another terminal, with Chrome installed:
python frontend/tests/browser_check.py
python frontend/tests/browser_polish.py
# Against a slow local API fixture:
python frontend/tests/browser_streaming.py
python frontend/tests/browser_navigation.py
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
