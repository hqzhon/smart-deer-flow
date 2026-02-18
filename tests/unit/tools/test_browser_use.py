import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.browser.browser_use import BrowserUseTool


class TestBrowserUseTool:
    def test_tool_properties(self):
        tool = BrowserUseTool()
        assert tool.name == "browser_use"
        assert "browser" in tool.category
        assert "browser" in tool.tags
        assert "action" in tool.required_parameters

    def test_parameters_schema(self):
        tool = BrowserUseTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "url" in params["properties"]
        assert "index" in params["properties"]
        assert "text" in params["properties"]

        action_enum = params["properties"]["action"]["enum"]
        assert "go_to_url" in action_enum
        assert "click_element" in action_enum
        assert "input_text" in action_enum
        assert "screenshot" in action_enum
        assert "get_state" in action_enum

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = BrowserUseTool()
        tool._initialized = True
        tool._page = AsyncMock()
        tool._context = AsyncMock()
        tool._context.pages = []

        result = await tool.async_execute(action="unknown_action")
        assert result.error is not None
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_go_to_url_missing_url(self):
        tool = BrowserUseTool()
        result = await tool._go_to_url(None)
        assert result.error is not None
        assert "URL is required" in result.error

    @pytest.mark.asyncio
    async def test_click_element_missing_params(self):
        tool = BrowserUseTool()
        result = await tool._click_element(None, None)
        assert result.error is not None
        assert "Either index or selector is required" in result.error

    @pytest.mark.asyncio
    async def test_input_text_missing_text(self):
        tool = BrowserUseTool()
        result = await tool._input_text(0, None, None)
        assert result.error is not None
        assert "Text is required" in result.error

    @pytest.mark.asyncio
    async def test_scroll_to_text_missing_text(self):
        tool = BrowserUseTool()
        result = await tool._scroll_to_text(None)
        assert result.error is not None
        assert "Text is required" in result.error

    @pytest.mark.asyncio
    async def test_send_keys_missing_keys(self):
        tool = BrowserUseTool()
        result = await tool._send_keys(None)
        assert result.error is not None
        assert "Keys are required" in result.error

    @pytest.mark.asyncio
    async def test_switch_tab_missing_tab_id(self):
        tool = BrowserUseTool()
        result = await tool._switch_tab(None)
        assert result.error is not None
        assert "Tab ID is required" in result.error

    @pytest.mark.asyncio
    async def test_open_tab_missing_url(self):
        tool = BrowserUseTool()
        result = await tool._open_tab(None)
        assert result.error is not None
        assert "URL is required" in result.error

    @pytest.mark.asyncio
    async def test_extract_content_missing_goal(self):
        tool = BrowserUseTool()
        result = await tool._extract_content(None)
        assert result.error is not None
        assert "Goal is required" in result.error

    @pytest.mark.asyncio
    async def test_select_dropdown_option_missing_text(self):
        tool = BrowserUseTool()
        result = await tool._select_dropdown_option(0, None, None)
        assert result.error is not None
        assert "Text is required" in result.error


class TestBrowserUseToolMocked:
    @pytest.fixture
    def mock_browser(self):
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.goto = AsyncMock()
        mock_page.go_back = AsyncMock()
        mock_page.reload = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)
        mock_page.get_by_text = MagicMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"fake_image_data")
        mock_page.close = AsyncMock()
        mock_page.bring_to_front = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.pages = [mock_page]
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        return {
            "browser": mock_browser,
            "context": mock_context,
            "page": mock_page,
        }

    @pytest.mark.asyncio
    async def test_go_to_url_success(self, mock_browser):
        tool = BrowserUseTool()

        tool._initialized = True
        tool._page = mock_browser["page"]
        tool._context = mock_browser["context"]

        result = await tool._go_to_url("https://example.com")

        assert result.output is not None
        assert "Navigated to" in result.output
        mock_browser["page"].goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot_success(self, mock_browser):
        tool = BrowserUseTool()

        tool._initialized = True
        tool._page = mock_browser["page"]
        tool._context = mock_browser["context"]

        result = await tool._screenshot()

        assert result.output is not None
        assert "Screenshot captured" in result.output

    @pytest.mark.asyncio
    async def test_get_state_success(self, mock_browser):
        tool = BrowserUseTool()

        with patch.object(
            tool,
            "_get_interactive_elements_info",
            new_callable=AsyncMock,
            return_value=[],
        ):
            tool._initialized = True
            tool._page = mock_browser["page"]
            tool._context = mock_browser["context"]

            result = await tool._get_state()

            assert result.output is not None

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_browser):
        tool = BrowserUseTool()
        tool._initialized = True
        tool._page = mock_browser["page"]
        tool._context = mock_browser["context"]
        tool._browser = mock_browser["browser"]

        await tool.cleanup()

        mock_browser["page"].close.assert_called_once()
        mock_browser["context"].close.assert_called_once()
        mock_browser["browser"].close.assert_called_once()
        assert tool._initialized is False
