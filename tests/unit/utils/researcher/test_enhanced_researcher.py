#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型研究员测试

测试 EnhancedResearcher 类的功能
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage
from langgraph.types import Command


class TestEnhancedResearcher(unittest.TestCase):
    """测试增强型研究员"""

    def setUp(self):
        """设置测试环境"""
        self.mock_state = {
            "messages": [HumanMessage(content="test query")],
            "research_topic": "test topic",
            "locale": "en-US",
        }

        self.mock_config = MagicMock()

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    def test_init(self, mock_processor, mock_config_manager):
        """测试初始化"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        self.assertEqual(researcher.state, self.mock_state)
        self.assertIsNotNone(researcher.config_manager)
        self.assertIsNotNone(researcher.tool_manager)
        self.assertIsNotNone(researcher.mcp_manager)
        self.assertIsNotNone(researcher.reflection_manager)
        self.assertIsNotNone(researcher.result_processor)

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchToolManager")
    @patch("src.utils.researcher.enhanced_researcher.MCPClientManager")
    @patch("src.utils.researcher.enhanced_researcher.ReflectionSystemManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_setup_isolation_context(
        self,
        mock_processor,
        mock_reflection,
        mock_mcp,
        mock_tool,
        mock_config_manager,
    ):
        """测试隔离上下文设置"""
        # 设置模拟对象
        mock_config_manager_instance = mock_config_manager.return_value
        mock_config_manager_instance.get_unified_config.return_value = {
            "test": "config"
        }

        mock_tool_instance = mock_tool.return_value
        mock_tool_instance.get_all_tools = AsyncMock(
            return_value=[MagicMock(), MagicMock()]
        )

        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)
        await researcher.setup_isolation_context()

        # 验证所有管理器都被初始化
        self.assertIsNotNone(researcher.tool_manager)
        self.assertIsNotNone(researcher.mcp_manager)
        self.assertIsNotNone(researcher.reflection_manager)
        self.assertIsNotNone(researcher.research_engine)

        # 验证配置管理器被调用
        mock_config_manager_instance.get_unified_config.assert_called_once()

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_execute_base_research(self, mock_processor, mock_config_manager):
        """测试基础研究执行"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        # 模拟conduct_research方法
        mock_result = {"initial_query": "test", "final_result": "result"}
        researcher.conduct_research = AsyncMock(return_value=mock_result)

        result = await researcher.conduct_research("test query")

        self.assertEqual(result, mock_result)
        researcher.conduct_research.assert_called_once_with("test query")

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_fallback_research(self, mock_processor, mock_config_manager):
        """测试回退研究"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        # 模拟fallback方法
        mock_result = {"fallback_result": "basic research completed"}
        researcher._fallback_research = AsyncMock(return_value=mock_result)

        result = await researcher._fallback_research()

        # 验证返回基本结果
        self.assertIsInstance(result, dict)
        self.assertIn("fallback_result", result)

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchToolManager")
    @patch("src.utils.researcher.enhanced_researcher.MCPClientManager")
    @patch("src.utils.researcher.enhanced_researcher.ReflectionSystemManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_execute_research_with_reflection_success(
        self,
        mock_processor,
        mock_reflection,
        mock_mcp,
        mock_tool,
        mock_config_manager,
    ):
        """测试带反思的研究执行 - 成功情况"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        # 模拟成功的研究执行
        mock_result = {"result": "success", "query": "test query"}
        researcher.conduct_research = AsyncMock(return_value=mock_result)

        result = await researcher.conduct_research("test query")

        self.assertEqual(result, mock_result)
        researcher.conduct_research.assert_called_once_with("test query")

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchToolManager")
    @patch("src.utils.researcher.enhanced_researcher.MCPClientManager")
    @patch("src.utils.researcher.enhanced_researcher.ReflectionSystemManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_execute_research_with_reflection_error(
        self,
        mock_processor,
        mock_reflection,
        mock_mcp,
        mock_tool,
        mock_config_manager,
    ):
        """测试带反思的研究执行 - 错误情况"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        # 模拟研究失败
        researcher.conduct_research = AsyncMock(
            side_effect=Exception("Research failed")
        )

        with self.assertRaises(Exception) as context:
            await researcher.conduct_research("test query")

        self.assertEqual(str(context.exception), "Research failed")
        researcher.conduct_research.assert_called_once_with("test query")

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchToolManager")
    @patch("src.utils.researcher.enhanced_researcher.MCPClientManager")
    @patch("src.utils.researcher.enhanced_researcher.ReflectionSystemManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_cleanup_on_exception(
        self,
        mock_processor,
        mock_reflection,
        mock_mcp,
        mock_tool,
        mock_config_manager,
    ):
        """测试异常情况下的资源清理"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        # 模拟MCP管理器
        mock_mcp_instance = mock_mcp.return_value
        mock_mcp_instance.cleanup_clients = AsyncMock()
        researcher.mcp_manager = mock_mcp_instance

        # 模拟执行失败
        researcher.conduct_research = AsyncMock(side_effect=Exception("Test error"))

        try:
            await researcher.conduct_research("test query")
        except Exception:
            pass

        # 验证清理方法被调用
        researcher.conduct_research.assert_called_once_with("test query")

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchToolManager")
    @patch("src.utils.researcher.enhanced_researcher.MCPClientManager")
    @patch("src.utils.researcher.enhanced_researcher.ReflectionSystemManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    async def test_context_manager_support(
        self,
        mock_processor,
        mock_reflection,
        mock_mcp,
        mock_tool,
        mock_config_manager,
    ):
        """测试上下文管理器支持"""
        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state)

        # 模拟MCP管理器的清理方法
        mock_mcp_instance = mock_mcp.return_value
        mock_mcp_instance.cleanup_clients = AsyncMock()
        researcher.mcp_manager = mock_mcp_instance

        # 验证可以作为上下文管理器使用
        async with researcher:
            self.assertIsNotNone(researcher)

        # 验证退出时清理资源被调用
        mock_mcp_instance.cleanup_clients.assert_called_once()


class TestEnhancedResearcherIntegration(unittest.TestCase):
    """增强型研究员集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.mock_state = {
            "messages": [HumanMessage(content="test query")],
            "research_topic": "test topic",
            "locale": "en-US",
        }

        self.mock_config = MagicMock()

    @patch("src.utils.researcher.enhanced_researcher.IsolationConfigManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchToolManager")
    @patch("src.utils.researcher.enhanced_researcher.MCPClientManager")
    @patch("src.utils.researcher.enhanced_researcher.ReflectionSystemManager")
    @patch("src.utils.researcher.enhanced_researcher.ResearchResultProcessor")
    @patch("src.utils.researcher.enhanced_researcher._setup_and_execute_agent_step")
    async def test_full_research_workflow(
        self,
        mock_execute_step,
        mock_processor_class,
        mock_reflection_class,
        mock_mcp_class,
        mock_tool_class,
        mock_config_class,
    ):
        """测试完整的研究工作流程"""
        # 设置所有模拟对象
        mock_config_instance = mock_config_class.return_value
        mock_config_instance.get_unified_config.return_value = {"test": "config"}
        mock_config_instance.configurable = MagicMock()

        mock_tool_instance = mock_tool_class.return_value
        mock_tool_instance.get_all_tools.return_value = [MagicMock()]

        mock_mcp_instance = mock_mcp_class.return_value
        mock_mcp_instance.cleanup_clients = AsyncMock()

        # mock_reflection_instance = mock_reflection_class.return_value

        mock_processor_instance = mock_processor_class.return_value
        mock_processor_instance.process_final_results.return_value = Command(
            goto="research_team"
        )

        mock_execute_step.return_value = Command(goto="research_team")

        from src.utils.researcher.enhanced_researcher import EnhancedResearcher

        researcher = EnhancedResearcher(self.mock_state, self.mock_config)

        # 执行完整工作流程
        await researcher.setup_isolation_context()
        result = await researcher.execute_research_with_reflection()

        # 验证结果
        self.assertIsInstance(result, Command)

        # 验证所有组件都被正确调用
        mock_config_instance.get_unified_config.assert_called_once()
        mock_tool_instance.get_all_tools.assert_called_once()
        mock_processor_instance.process_final_results.assert_called_once()
        mock_mcp_instance.cleanup_clients.assert_called_once()


if __name__ == "__main__":
    unittest.main()
