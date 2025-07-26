#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带隔离的研究员节点集成测试

测试简化后的 researcher_node_with_isolation 函数的集成功能
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage
from langgraph.types import Command


class TestResearcherNodeWithIsolation(unittest.TestCase):
    """测试带隔离的研究员节点"""

    def setUp(self):
        """设置测试环境"""
        self.mock_state = {
            "messages": [HumanMessage(content="test research query")],
            "research_topic": "AI and machine learning trends",
            "locale": "en-US",
            "plan": {
                "steps": [{"description": "Research AI trends", "type": "research"}]
            },
        }

        self.mock_config = MagicMock()

    @patch("src.graph.nodes.EnhancedResearcher")
    async def test_researcher_node_with_isolation_success(
        self, mock_enhanced_researcher_class
    ):
        """测试成功的研究员节点执行"""
        # 设置模拟的EnhancedResearcher
        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock()
        mock_researcher.execute_research_with_reflection = AsyncMock(
            return_value=Command(goto="research_team")
        )

        from src.graph.nodes import researcher_node_with_isolation

        result = await researcher_node_with_isolation(self.mock_state, self.mock_config)

        # 验证结果
        self.assertIsInstance(result, Command)
        self.assertEqual(result.goto, "research_team")

        # 验证EnhancedResearcher被正确调用
        mock_enhanced_researcher_class.assert_called_once_with(
            self.mock_state, self.mock_config
        )
        mock_researcher.setup_isolation_context.assert_called_once()
        mock_researcher.execute_research_with_reflection.assert_called_once()

    @patch("src.graph.nodes.EnhancedResearcher")
    @patch("src.graph.nodes.researcher_node")
    async def test_researcher_node_with_isolation_fallback(
        self, mock_researcher_node, mock_enhanced_researcher_class
    ):
        """测试研究员节点的降级处理"""
        # 设置EnhancedResearcher失败
        mock_enhanced_researcher_class.side_effect = Exception(
            "Enhanced researcher failed"
        )

        # 设置降级节点返回值
        mock_researcher_node.return_value = Command(goto="research_team")

        from src.graph.nodes import researcher_node_with_isolation

        result = await researcher_node_with_isolation(self.mock_state, self.mock_config)

        # 验证降级到标准研究员节点
        self.assertIsInstance(result, Command)
        mock_researcher_node.assert_called_once_with(self.mock_state, self.mock_config)

    @patch("src.graph.nodes.EnhancedResearcher")
    @patch("src.graph.nodes.researcher_node")
    async def test_researcher_node_setup_failure_fallback(
        self, mock_researcher_node, mock_enhanced_researcher_class
    ):
        """测试设置失败时的降级处理"""
        # 设置EnhancedResearcher创建成功但设置失败
        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock(
            side_effect=Exception("Setup failed")
        )

        # 设置降级节点返回值
        mock_researcher_node.return_value = Command(goto="research_team")

        from src.graph.nodes import researcher_node_with_isolation

        result = await researcher_node_with_isolation(self.mock_state, self.mock_config)

        # 验证降级到标准研究员节点
        self.assertIsInstance(result, Command)
        mock_researcher_node.assert_called_once_with(self.mock_state, self.mock_config)

    @patch("src.graph.nodes.EnhancedResearcher")
    @patch("src.graph.nodes.researcher_node")
    async def test_researcher_node_execution_failure_fallback(
        self, mock_researcher_node, mock_enhanced_researcher_class
    ):
        """测试执行失败时的降级处理"""
        # 设置EnhancedResearcher设置成功但执行失败
        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock()
        mock_researcher.execute_research_with_reflection = AsyncMock(
            side_effect=Exception("Execution failed")
        )

        # 设置降级节点返回值
        mock_researcher_node.return_value = Command(goto="research_team")

        from src.graph.nodes import researcher_node_with_isolation

        result = await researcher_node_with_isolation(self.mock_state, self.mock_config)

        # 验证降级到标准研究员节点
        self.assertIsInstance(result, Command)
        mock_researcher_node.assert_called_once_with(self.mock_state, self.mock_config)

    @patch("src.graph.nodes.EnhancedResearcher")
    async def test_researcher_node_with_different_state_types(
        self, mock_enhanced_researcher_class
    ):
        """测试不同状态类型的处理"""
        # 测试不同的状态配置
        test_states = [
            {
                "messages": [HumanMessage(content="simple query")],
                "research_topic": "simple topic",
            },
            {
                "messages": [HumanMessage(content="complex query with multiple parts")],
                "research_topic": "complex topic",
                "locale": "zh-CN",
                "plan": {"steps": []},
            },
        ]

        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock()
        mock_researcher.execute_research_with_reflection = AsyncMock(
            return_value=Command(goto="research_team")
        )

        from src.graph.nodes import researcher_node_with_isolation

        for state in test_states:
            with self.subTest(state=state):
                result = await researcher_node_with_isolation(state, self.mock_config)
                self.assertIsInstance(result, Command)

    @patch("src.graph.nodes.EnhancedResearcher")
    async def test_researcher_node_logging(self, mock_enhanced_researcher_class):
        """测试日志记录功能"""
        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock()
        mock_researcher.execute_research_with_reflection = AsyncMock(
            return_value=Command(goto="research_team")
        )

        with patch("src.graph.nodes.logger") as mock_logger:
            from src.graph.nodes import researcher_node_with_isolation

            await researcher_node_with_isolation(self.mock_state, self.mock_config)

            # 验证日志记录
            mock_logger.info.assert_called()
            # 检查是否记录了开始和完成的日志
            info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            self.assertTrue(any("started" in call for call in info_calls))
            self.assertTrue(any("completed" in call for call in info_calls))

    @patch("src.graph.nodes.EnhancedResearcher")
    @patch("src.graph.nodes.researcher_node")
    async def test_researcher_node_error_logging(
        self, mock_researcher_node, mock_enhanced_researcher_class
    ):
        """测试错误日志记录"""
        mock_enhanced_researcher_class.side_effect = Exception("Test error")
        mock_researcher_node.return_value = Command(goto="research_team")

        with patch("src.graph.nodes.logger") as mock_logger:
            from src.graph.nodes import researcher_node_with_isolation

            await researcher_node_with_isolation(self.mock_state, self.mock_config)

            # 验证错误日志记录
            mock_logger.error.assert_called()
            mock_logger.warning.assert_called()


class TestResearcherNodePerformance(unittest.TestCase):
    """研究员节点性能测试"""

    def setUp(self):
        """设置测试环境"""
        self.mock_state = {
            "messages": [HumanMessage(content="performance test query")],
            "research_topic": "performance testing",
        }

        self.mock_config = MagicMock()

    @patch("src.graph.nodes.EnhancedResearcher")
    async def test_researcher_node_execution_time(self, mock_enhanced_researcher_class):
        """测试研究员节点执行时间"""
        import time

        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock()
        mock_researcher.execute_research_with_reflection = AsyncMock(
            return_value=Command(goto="research_team")
        )

        from src.graph.nodes import researcher_node_with_isolation

        start_time = time.time()
        result = await researcher_node_with_isolation(self.mock_state, self.mock_config)
        execution_time = time.time() - start_time

        # 验证执行时间合理（应该很快，因为都是模拟）
        self.assertLess(execution_time, 1.0)  # 应该在1秒内完成
        self.assertIsInstance(result, Command)

    @patch("src.graph.nodes.EnhancedResearcher")
    async def test_researcher_node_memory_usage(self, mock_enhanced_researcher_class):
        """测试研究员节点内存使用"""
        import gc

        mock_researcher = mock_enhanced_researcher_class.return_value
        mock_researcher.setup_isolation_context = AsyncMock()
        mock_researcher.execute_research_with_reflection = AsyncMock(
            return_value=Command(goto="research_team")
        )

        from src.graph.nodes import researcher_node_with_isolation

        # 强制垃圾回收
        gc.collect()
        initial_objects = len(gc.get_objects())

        # 执行多次以检查内存泄漏
        for _ in range(10):
            await researcher_node_with_isolation(self.mock_state, self.mock_config)

        gc.collect()
        final_objects = len(gc.get_objects())

        # 验证没有显著的内存泄漏
        object_increase = final_objects - initial_objects
        self.assertLess(object_increase, 1000)  # 允许一些合理的对象增长


if __name__ == "__main__":
    unittest.main()
