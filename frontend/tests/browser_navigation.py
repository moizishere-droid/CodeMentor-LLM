"""Stress header click bursts and verify history, budgets, continuation, caret."""

import os

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.getenv("CHROME_PATH", "C:/Program Files/Google/Chrome/Application/chrome.exe"),
            headless=True,
        )
        for width in (360, 768, 1440):
            page = browser.new_page(viewport={"width": width, "height": 960})
            page.goto(os.getenv("UI_URL", "http://localhost:8506"))
            header = page.frame_locator('iframe[title="ui.codementor_header"]')
            expect(header.locator("nav")).to_be_visible()
            frame = next(frame for frame in page.frames if "/component/" in frame.url)
            # Re-clicking the active page must generate no component messages.
            emitted = frame.evaluate("""async () => {
              let count = 0;
              const original = parent.postMessage;
              parent.postMessage = function(message, ...args) {
                if (message.type === 'streamlit:setComponentValue') count++;
                return original.call(parent, message, ...args);
              };
              for (let i=0; i<30; i++) document.getElementById('workspace').click();
              await new Promise(resolve => setTimeout(resolve, 150));
              parent.postMessage = original;
              return count;
            }""")
            assert emitted == 0
            for _cycle in range(3):
                frame.evaluate("""() => {
                  for(let i=0;i<15;i++) {
                    document.getElementById('logs').click();
                    document.getElementById('workspace').click();
                  }
                  document.getElementById('logs').click();
                }""")
                expect(page.get_by_role("heading", name="Request log", exact=True)).to_be_visible()
                expect(header.locator("nav")).to_be_visible()
                expect(header.get_by_role("button", name="Request Log")).to_have_attribute("aria-current", "page")
                frame.evaluate("""() => {
                  for(let i=0;i<15;i++) {
                    document.getElementById('workspace').click();
                    document.getElementById('logs').click();
                  }
                  document.getElementById('workspace').click();
                }""")
                expect(page.get_by_role("heading", name="Let’s work it out, together.")).to_be_visible()
                expect(header.locator("nav")).to_be_visible()

            textarea = page.get_by_role("textbox", name="Your coding question")
            textarea.focus()
            expect(textarea).to_have_css("caret-color", "rgb(32, 36, 38)")
            page.get_by_text("Detailed code", exact=True).click()
            textarea.fill("Write a complete function.")
            page.get_by_role("button", name="Send question").click()
            expect(page.get_by_text("End of fixture response.", exact=True)).to_be_attached(timeout=15000)
            expect(textarea).to_have_css("caret-color", "rgb(241, 243, 242)")
            page.get_by_role("button", name="Continue response", exact=True).click()
            expect(page.get_by_text("End of continuation.", exact=True)).to_be_attached(timeout=15000)
            expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(1)
            expect(page.get_by_role("radio", name="Detailed code", exact=True)).to_be_checked()
            expect(header.locator("nav")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            print(width, "rapid navigation, persistent header, response mode, continuation and caret passed")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
