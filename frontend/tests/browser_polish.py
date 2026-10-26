"""Check synchronized theme updates and send tooltip contrast in real Chrome."""

import os
import re

from playwright.sync_api import expect, sync_playwright


def contrast(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [float(value) / 255 for value in re.findall(r"[\d.]+", color)[:3]]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.getenv(
                "CHROME_PATH", "C:/Program Files/Google/Chrome/Application/chrome.exe"
            ),
            headless=True,
        )
        for width in (360, 768, 1440):
            page = browser.new_page(viewport={"width": width, "height": 960})
            page.goto(os.getenv("UI_URL", "http://localhost:8501"))
            frame = page.frame_locator('iframe[title="ui.codementor_header"]')
            expect(frame.get_by_role("button", name="Appearance")).to_be_visible()
            for revision, appearance in enumerate(("dark", "light", "dark", "light"), start=1):
                result = frame.locator("#appearance").evaluate(
                    """button => {
                      const previous = document.documentElement.style.colorScheme;
                      button.click();
                      window.dispatchEvent(new MessageEvent('message', {
                        source: parent,
                        data: {type: 'streamlit:render', args: {
                          page: 'workspace', appearance: previous, ack_revision: 0
                        }}
                      }));
                      return {
                        header: document.documentElement.style.colorScheme,
                        body: parent.document.documentElement.style.colorScheme,
                        page: parent.document.documentElement.style.getPropertyValue('--cm-appearance')
                      };
                    }"""
                )
                assert result == {"header": appearance, "body": appearance, "page": appearance}, result
                expect(page.locator('[data-testid="stAppViewContainer"]')).to_have_css(
                    "background-color", "rgb(20, 21, 22)" if appearance == "dark" else "rgb(255, 255, 255)"
                )
                expect(frame.locator("html")).to_have_css("color-scheme", appearance)
                expect(frame.locator("html")).to_have_attribute("data-revision", str(revision))
                if os.getenv("UI_POLISH_FIXTURE") == "1":
                    inline = page.locator('[data-testid="stChatMessage"] p > code')
                    expect(inline).to_have_count(3)
                    expect(inline.first).to_have_css("background-color", "rgba(0, 0, 0, 0)")
                    expect(page.locator('[data-testid="stChatMessage"] pre')).to_be_visible()
                    expect(page.locator(".cm-heading")).to_have_count(0)
                    answer_box = page.locator('[data-testid="stChatMessage"]').bounding_box()
                    composer_box = page.locator('[data-testid="stForm"]').bounding_box()
                    assert answer_box and composer_box and answer_box["y"] < composer_box["y"]
                send = page.get_by_role("button", name="Send question")
                send.hover()
                if width < 768:
                    continue  # Native Streamlit omits hover help on compact layouts.
                tooltip = page.locator('[data-baseweb="tooltip"]')
                expect(tooltip).to_be_visible()
                styles = tooltip.evaluate(
                    """element => {
                      const text = element.querySelector('[data-testid="stMarkdownContainer"]');
                      let surface = text;
                      while (surface.parentElement && getComputedStyle(surface).backgroundColor === 'rgba(0, 0, 0, 0)') surface = surface.parentElement;
                      return {background: getComputedStyle(surface).backgroundColor,
                              text: getComputedStyle(text).color};
                    }"""
                )
                print(width, appearance, styles)
                assert contrast(styles["text"], styles["background"]) >= 4.5, styles
                page.mouse.move(0, 0)
            page.emulate_media(reduced_motion="reduce")
            expect(page.locator('[data-testid="stAppViewContainer"]')).to_have_css("transition-duration", "0s")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
