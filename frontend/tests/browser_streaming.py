"""Run against the slow local fixture to catch horizontal streaming jitter."""

import os

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.getenv("CHROME_PATH", "C:/Program Files/Google/Chrome/Application/chrome.exe"),
            headless=True,
        )
        for width in (360, 768, 1440):
            for appearance in ("light", "dark"):
                page = browser.new_page(viewport={"width": width, "height": 960})
                page.goto(os.getenv("UI_URL", "http://localhost:8506"))
                header = page.frame_locator('iframe[title="ui.codementor_header"]')
                expect(header.get_by_role("button", name="Appearance")).to_be_visible()
                if appearance == "dark":
                    header.get_by_role("button", name="Appearance").click()
                    expect(header.locator("html")).to_have_attribute("data-revision", "1")
                page.get_by_role("textbox", name="Your coding question").fill("Explain a circular linked list.")
                page.get_by_role("button", name="Send question").click()
                expect(page.locator('[data-testid="stChatMessage"]')).to_be_visible()
                page.evaluate("""() => {
                  window.cmSamples = [];
                  window.cmRecording = true;
                  const sample = () => {
                    const response = document.querySelector('[data-testid="stChatMessage"]');
                    if (response) {
                      const bounds = response.getBoundingClientRect();
                      const main = document.querySelector('section.main');
                      window.cmSamples.push({x: bounds.x, width: bounds.width, scroll: main.scrollTop});
                    }
                    if (window.cmRecording) requestAnimationFrame(sample);
                  };
                  requestAnimationFrame(sample);
                }""")
                expect(page.get_by_text("End of fixture response.", exact=True)).to_be_attached(timeout=15000)
                samples = page.evaluate("() => { window.cmRecording = false; return window.cmSamples; }")
                assert len(samples) > 5
                for key in ("x", "width", "scroll"):
                    drift = max(sample[key] for sample in samples) - min(sample[key] for sample in samples)
                    assert drift <= 1, (width, appearance, key, drift)
                heading = page.locator('[data-testid="stChatMessage"] h1')
                expect(heading).to_have_css("color", "rgb(32, 36, 38)" if appearance == "light" else "rgb(241, 243, 242)")
                form = page.locator('[data-testid="stForm"]')
                expect(form).to_have_css("background-color", "rgb(32, 36, 38)")
                box = form.bounding_box()
                assert box and box["height"] <= 110, box
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                print(width, appearance, "stable streaming; compact dark composer; readable heading")
                page.close()
        browser.close()


if __name__ == "__main__":
    main()
