"""Run against a local Streamlit server; uses the installed Chrome browser.

Optional developer dependency: playwright. Screenshots go to .ui-checks/.
"""

import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUTPUT = Path(".ui-checks")
OUTPUT.mkdir(exist_ok=True)
CHROME = os.getenv("CHROME_PATH", "C:/Program Files/Google/Chrome/Application/chrome.exe")


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
        results = []
        for width in (360, 768, 1440):
            page = browser.new_page(viewport={"width": width, "height": 960})
            page.goto(os.getenv("UI_URL", "http://localhost:8501"))
            header = page.frame_locator('iframe[title="ui.codementor_header"]')
            expect(header.get_by_role("button", name="Appearance", exact=True)).to_be_visible()
            for appearance in ("light", "dark"):
                if appearance == "dark":
                    header.get_by_role("button", name="Appearance", exact=True).click()
                    expect(page.locator('[data-testid="stAppViewContainer"]')).to_have_css(
                        "background-color", "rgb(20, 21, 22)"
                    )
                expect(page.get_by_role("heading", name="Let’s work it out, together.")).to_be_visible()
                expect(page.locator('[data-testid="stException"]')).to_have_count(0)
                dimensions = header.locator("body").evaluate(
                    "el => ({width: innerWidth, content: el.scrollWidth})"
                )
                assert dimensions["content"] <= dimensions["width"], dimensions
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                form = page.locator('[data-testid="stForm"]').bounding_box()
                assert form and form["width"] >= min(width - 48, 600)
                theme = header.get_by_role("button", name="Appearance", exact=True)
                box = theme.bounding_box()
                assert box and box["width"] >= 40 and box["height"] >= 40
                page.keyboard.press("Tab")
                theme.focus()
                expect(header.locator("#appearance-tip")).to_be_visible()
                expect(theme).to_have_css("outline-style", "solid")
                page.screenshot(path=str(OUTPUT / f"workspace-{width}-{appearance}.png"), full_page=True, animations="disabled")
                header.get_by_role("button", name="Request Log", exact=True).click()
                expect(page.get_by_role("heading", name="Request log", exact=True)).to_be_visible()
                expect(header.get_by_role("button", name="Request Log", exact=True)).to_have_attribute(
                    "aria-current", "page"
                )
                expect(page.locator('[data-testid="stException"]')).to_have_count(0)
                expect(page.locator('[data-testid="stAlert"], .cm-log-header').first).to_be_visible(timeout=45000)
                if os.getenv("UI_FIXTURE") == "1":
                    expect(page.locator(".cm-status-success")).to_have_text("Success")
                    expect(page.locator(".cm-status-failed")).to_have_text("Failed")
                    expect(page.locator('[data-testid="stDataFrame"]')).to_have_count(0)
                    expect(page.locator('[data-testid="stExpander"]')).to_have_count(0)
                    assert page.locator('pre').evaluate_all(
                        "elements => elements.every(element => element.scrollHeight <= element.clientHeight + 1)"
                    )
                page.screenshot(path=str(OUTPUT / f"logs-{width}-{appearance}.png"), full_page=True, animations="disabled")
                header.get_by_role("button", name="Workspace", exact=True).click()
                expect(page.get_by_role("heading", name="Let’s work it out, together.")).to_be_visible()
                results.append({"width": width, "appearance": appearance, "status": "passed"})
            if os.getenv("UI_FIXTURE") == "1":
                page.get_by_role("button", name="Build something", exact=True).click()
                expect(page.get_by_role("textbox", name="Your coding question")).to_have_value(
                    "Write a Python function to reverse a string."
                )
                page.get_by_role("button", name="Send question").click()
                expect(page.locator('[data-testid="stChatMessage"] pre code')).to_contain_text("reverse_string")
                expect(page.get_by_text("End of fixture response.", exact=True)).to_be_visible(timeout=15000)
                expect(page.locator(".cm-heading")).to_have_count(0)
                expect(page.get_by_role("button", name="Build something", exact=True)).to_have_count(0)
                expect(page.get_by_role("textbox", name="Your coding question")).to_have_value("")
                response_box = page.locator('[data-testid="stChatMessage"]').bounding_box()
                composer_box = page.locator('[data-testid="stForm"]').bounding_box()
                assert response_box and composer_box and response_box["y"] < composer_box["y"]
                header.get_by_role("button", name="Appearance", exact=True).click()
                expect(page.locator('[data-testid="stChatMessage"] pre code')).to_contain_text("reverse_string")
                expect(page.locator('[data-testid="stAppViewContainer"]')).to_have_css("background-color", "rgb(255, 255, 255)")
                expect(page.locator('[data-stale="true"]')).to_have_count(0)
                expect(page.get_by_role("textbox", name="Your coding question")).to_have_value("")
                page.screenshot(path=str(OUTPUT / f"response-{width}.png"), full_page=True, animations="disabled")
            page.close()
        browser.close()
        (OUTPUT / "browser-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
