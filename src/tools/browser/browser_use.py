import asyncio
import base64
import json
import logging
from typing import Any, Optional

from src.tools.base_tool import BaseTool
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)

_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs
* Screenshots: Capture the current page state

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""


class BrowserUseTool(BaseTool):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION

    def __init__(self, **data):
        super().__init__(**data)
        self._lock = asyncio.Lock()
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "go_to_url",
                        "click_element",
                        "input_text",
                        "scroll_down",
                        "scroll_up",
                        "scroll_to_text",
                        "send_keys",
                        "get_dropdown_options",
                        "select_dropdown_option",
                        "go_back",
                        "refresh",
                        "wait",
                        "extract_content",
                        "switch_tab",
                        "open_tab",
                        "close_tab",
                        "screenshot",
                        "get_state",
                    ],
                    "description": "The browser action to perform",
                },
                "url": {
                    "type": "string",
                    "description": "URL for 'go_to_url' or 'open_tab' actions",
                },
                "index": {
                    "type": "integer",
                    "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
                },
                "text": {
                    "type": "string",
                    "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
                },
                "scroll_amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
                },
                "tab_id": {
                    "type": "integer",
                    "description": "Tab ID for 'switch_tab' action",
                },
                "goal": {
                    "type": "string",
                    "description": "Extraction goal for 'extract_content' action",
                },
                "keys": {
                    "type": "string",
                    "description": "Keys to send for 'send_keys' action",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Seconds to wait for 'wait' action",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for element targeting",
                },
            },
            "required": ["action"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["action"]

    @property
    def category(self) -> str:
        return "browser"

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "web", "scraping"]

    async def _ensure_browser_initialized(self) -> None:
        if self._initialized and self._page:
            return

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            self._page = await self._context.new_page()
            self._initialized = True
            logger.info("Browser initialized successfully")
        except ImportError:
            raise RuntimeError(
                "playwright is not installed. Install it with: pip install playwright && playwright install chromium"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize browser: {e}")

    def execute(self, **kwargs) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.async_execute(**kwargs))

    async def async_execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        selector: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        async with self.lock:
            try:
                await self._ensure_browser_initialized()

                if action == "go_to_url":
                    return await self._go_to_url(url)
                elif action == "go_back":
                    return await self._go_back()
                elif action == "refresh":
                    return await self._refresh()
                elif action == "click_element":
                    return await self._click_element(index, selector)
                elif action == "input_text":
                    return await self._input_text(index, text, selector)
                elif action == "scroll_down":
                    return await self._scroll(1, scroll_amount)
                elif action == "scroll_up":
                    return await self._scroll(-1, scroll_amount)
                elif action == "scroll_to_text":
                    return await self._scroll_to_text(text)
                elif action == "send_keys":
                    return await self._send_keys(keys)
                elif action == "get_dropdown_options":
                    return await self._get_dropdown_options(index, selector)
                elif action == "select_dropdown_option":
                    return await self._select_dropdown_option(index, text, selector)
                elif action == "extract_content":
                    return await self._extract_content(goal)
                elif action == "switch_tab":
                    return await self._switch_tab(tab_id)
                elif action == "open_tab":
                    return await self._open_tab(url)
                elif action == "close_tab":
                    return await self._close_tab()
                elif action == "screenshot":
                    return await self._screenshot()
                elif action == "get_state":
                    return await self._get_state()
                elif action == "wait":
                    return await self._wait(seconds)
                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                logger.error(f"Browser action '{action}' failed: {e}")
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def _go_to_url(self, url: Optional[str]) -> ToolResult:
        if not url:
            return ToolResult(error="URL is required for 'go_to_url' action")
        await self._page.goto(url, wait_until="networkidle")
        return ToolResult(output=f"Navigated to {url}")

    async def _go_back(self) -> ToolResult:
        await self._page.go_back(wait_until="networkidle")
        return ToolResult(output="Navigated back")

    async def _refresh(self) -> ToolResult:
        await self._page.reload(wait_until="networkidle")
        return ToolResult(output="Page refreshed")

    async def _click_element(
        self, index: Optional[int], selector: Optional[str]
    ) -> ToolResult:
        if selector:
            await self._page.click(selector)
            return ToolResult(output=f"Clicked element with selector: {selector}")
        if index is not None:
            elements = await self._get_interactive_elements()
            if index < 0 or index >= len(elements):
                return ToolResult(error=f"Element index {index} out of range")
            await elements[index].click()
            return ToolResult(output=f"Clicked element at index {index}")
        return ToolResult(
            error="Either index or selector is required for 'click_element' action"
        )

    async def _input_text(
        self, index: Optional[int], text: Optional[str], selector: Optional[str]
    ) -> ToolResult:
        if not text:
            return ToolResult(error="Text is required for 'input_text' action")
        if selector:
            await self._page.fill(selector, text)
            return ToolResult(
                output=f"Input '{text}' into element with selector: {selector}"
            )
        if index is not None:
            elements = await self._get_interactive_elements()
            if index < 0 or index >= len(elements):
                return ToolResult(error=f"Element index {index} out of range")
            await elements[index].fill(text)
            return ToolResult(output=f"Input '{text}' into element at index {index}")
        return ToolResult(
            error="Either index or selector is required for 'input_text' action"
        )

    async def _scroll(self, direction: int, amount: Optional[int]) -> ToolResult:
        scroll_pixels = amount if amount else 500
        await self._page.evaluate(f"window.scrollBy(0, {direction * scroll_pixels})")
        direction_text = "down" if direction > 0 else "up"
        return ToolResult(output=f"Scrolled {direction_text} by {scroll_pixels} pixels")

    async def _scroll_to_text(self, text: Optional[str]) -> ToolResult:
        if not text:
            return ToolResult(error="Text is required for 'scroll_to_text' action")
        locator = self._page.get_by_text(text, exact=False)
        await locator.scroll_into_view_if_needed()
        return ToolResult(output=f"Scrolled to text: '{text}'")

    async def _send_keys(self, keys: Optional[str]) -> ToolResult:
        if not keys:
            return ToolResult(error="Keys are required for 'send_keys' action")
        await self._page.keyboard.press(keys)
        return ToolResult(output=f"Sent keys: {keys}")

    async def _get_dropdown_options(
        self, index: Optional[int], selector: Optional[str]
    ) -> ToolResult:
        target_selector = selector
        if not target_selector and index is not None:
            elements = await self._get_interactive_elements()
            if index < 0 or index >= len(elements):
                return ToolResult(error=f"Element index {index} out of range")
            target_selector = f"select:nth-of-type({index + 1})"

        if not target_selector:
            return ToolResult(error="Either index or selector is required")

        options = await self._page.evaluate(f"""
            () => {{
                const select = document.querySelector('{target_selector}');
                if (!select) return null;
                return Array.from(select.options).map(opt => ({{
                    text: opt.text,
                    value: opt.value,
                    index: opt.index
                }}));
            }}
        """)
        return ToolResult(output=f"Dropdown options: {json.dumps(options, indent=2)}")

    async def _select_dropdown_option(
        self, index: Optional[int], text: Optional[str], selector: Optional[str]
    ) -> ToolResult:
        if not text:
            return ToolResult(
                error="Text is required for 'select_dropdown_option' action"
            )

        target_selector = selector
        if not target_selector and index is not None:
            target_selector = f"select:nth-of-type({index + 1})"

        if not target_selector:
            return ToolResult(error="Either index or selector is required")

        await self._page.select_option(target_selector, label=text)
        return ToolResult(output=f"Selected option '{text}' from dropdown")

    async def _extract_content(self, goal: Optional[str]) -> ToolResult:
        if not goal:
            return ToolResult(error="Goal is required for 'extract_content' action")

        content = await self._page.content()

        import markdownify

        markdown_content = markdownify.markdownify(content)

        max_length = 5000
        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length] + "..."

        return ToolResult(
            output=f"Extracted content (goal: {goal}):\n{markdown_content}"
        )

    async def _switch_tab(self, tab_id: Optional[int]) -> ToolResult:
        if tab_id is None:
            return ToolResult(error="Tab ID is required for 'switch_tab' action")
        pages = self._context.pages
        if tab_id < 0 or tab_id >= len(pages):
            return ToolResult(error=f"Tab ID {tab_id} out of range")
        self._page = pages[tab_id]
        await self._page.bring_to_front()
        return ToolResult(output=f"Switched to tab {tab_id}")

    async def _open_tab(self, url: Optional[str]) -> ToolResult:
        if not url:
            return ToolResult(error="URL is required for 'open_tab' action")
        new_page = await self._context.new_page()
        await new_page.goto(url, wait_until="networkidle")
        self._page = new_page
        return ToolResult(output=f"Opened new tab with {url}")

    async def _close_tab(self) -> ToolResult:
        pages = self._context.pages
        if len(pages) <= 1:
            return ToolResult(error="Cannot close the last tab")
        await self._page.close()
        self._page = pages[-2] if pages[-1] == self._page else pages[-1]
        return ToolResult(output="Closed current tab")

    async def _screenshot(self) -> ToolResult:
        screenshot_bytes = await self._page.screenshot(
            full_page=False, type="jpeg", quality=80
        )
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return ToolResult(
            output="Screenshot captured",
            base64_image=screenshot_base64,
        )

    async def _get_state(self) -> ToolResult:
        title = await self._page.title()
        url = self._page.url
        tabs = [
            {"id": i, "url": p.url, "title": await p.title()}
            for i, p in enumerate(self._context.pages)
        ]

        interactive_elements = await self._get_interactive_elements_info()

        state_info = {
            "url": url,
            "title": title,
            "tabs": tabs,
            "interactive_elements": interactive_elements,
        }
        return ToolResult(output=json.dumps(state_info, indent=2, ensure_ascii=False))

    async def _wait(self, seconds: Optional[int]) -> ToolResult:
        wait_seconds = seconds if seconds is not None else 3
        await asyncio.sleep(wait_seconds)
        return ToolResult(output=f"Waited for {wait_seconds} seconds")

    async def _get_interactive_elements(self) -> list:
        return await self._page.locator(
            "a, button, input, select, textarea, [onclick], [role='button']"
        ).all()

    async def _get_interactive_elements_info(self) -> list[dict]:
        elements = await self._get_interactive_elements()
        elements_info = []
        for i, element in enumerate(elements[:50]):
            try:
                tag = await element.evaluate("el => el.tagName.toLowerCase()")
                text = await element.inner_text()
                elements_info.append(
                    {
                        "index": i,
                        "tag": tag,
                        "text": text[:100] if text else "",
                    }
                )
            except Exception:
                continue
        return elements_info

    async def cleanup(self) -> None:
        async with self.lock:
            if self._page:
                await self._page.close()
                self._page = None
            if self._context:
                await self._context.close()
                self._context = None
            if self._browser:
                await self._browser.close()
                self._browser = None
            if hasattr(self, "_playwright") and self._playwright:
                await self._playwright.stop()
                self._playwright = None
            self._initialized = False

    def __del__(self):
        if self._initialized:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
            except Exception:
                pass
