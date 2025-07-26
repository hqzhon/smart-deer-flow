#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔离配置管理器测试

测试 IsolationConfigManager 类的功能
"""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage


class TestIsolationConfigManager(unittest.TestCase):
    """测试隔离配置管理器"""

    def setUp(self):
        """设置测试环境"""
        self.mock_state = {
            "messages": [HumanMessage(content="test query")],
            "research_topic": "test topic",
            "locale": "en-US",
        }

        self.mock_config = MagicMock()
        self.mock_configurable = MagicMock()
        self.mock_configurable.agents.max_search_results = 5
        self.mock_configurable.agents.enable_deep_thinking = True
        self.mock_configurable.content.enable_smart_filtering = True

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_init(self, mock_get_config):
        """测试初始化"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        manager = IsolationConfigManager(self.mock_state, self.mock_config)

        self.assertEqual(manager.state, self.mock_state)
        self.assertEqual(manager.config, self.mock_config)
        self.assertEqual(manager.configurable, self.mock_configurable)
        mock_get_config.assert_called_once_with(self.mock_config)

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_setup_language_config(self, mock_get_config):
        """测试语言配置设置"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        manager = IsolationConfigManager(self.mock_state, self.mock_config)
        language_config = manager.setup_language_config()

        self.assertIsInstance(language_config, dict)
        self.assertIn("locale", language_config)
        self.assertEqual(language_config["locale"], "en-US")

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_setup_isolation_level(self, mock_get_config):
        """测试隔离级别配置"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import (
            IsolationConfigManager,
            ResearcherContextConfig,
        )

        manager = IsolationConfigManager(self.mock_state, self.mock_config)
        isolation_config = manager.setup_isolation_level()

        self.assertIsInstance(isolation_config, ResearcherContextConfig)
        self.assertTrue(hasattr(isolation_config, "max_context_steps"))
        self.assertTrue(hasattr(isolation_config, "isolation_level"))

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_setup_metrics_config(self, mock_get_config):
        """测试度量配置设置"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        manager = IsolationConfigManager(self.mock_state, self.mock_config)
        config = manager.setup_phase3_config()

        # 检查必需的配置项
        self.assertIn("researcher_isolation_metrics", config)
        self.assertIn("researcher_isolation_threshold", config)

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_setup_reflection_config(self, mock_get_config):
        """测试反射配置设置"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        manager = IsolationConfigManager(self.mock_state, self.mock_config)
        reflection_config = manager.setup_reflection_config()

        # 检查必需的配置项
        self.assertIsInstance(reflection_config, dict)
        self.assertIn("enable_enhanced_reflection", reflection_config)
        self.assertIn("max_reflection_loops", reflection_config)
        self.assertIn("reflection_model", reflection_config)

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_get_unified_config(self, mock_get_config):
        """测试统一配置获取"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import (
            IsolationConfigManager,
            UnifiedResearchConfig,
        )

        manager = IsolationConfigManager(self.mock_state, self.mock_config)
        unified_config = manager.get_unified_config()

        self.assertIsInstance(unified_config, UnifiedResearchConfig)
        # 验证包含所有配置属性
        self.assertTrue(hasattr(unified_config, "locale"))
        self.assertTrue(hasattr(unified_config, "isolation_config"))
        self.assertTrue(hasattr(unified_config, "enable_enhanced_reflection"))
        self.assertTrue(hasattr(unified_config, "max_search_results"))
        self.assertTrue(hasattr(unified_config, "mcp_enabled"))

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_config_validation(self, mock_get_config):
        """测试配置验证"""
        mock_get_config.return_value = self.mock_configurable

        from src.utils.researcher.isolation_config_manager import (
            IsolationConfigManager,
            UnifiedResearchConfig,
        )

        manager = IsolationConfigManager(self.mock_state, self.mock_config)

        # 测试获取统一配置
        config = manager.get_unified_config()

        # 验证配置结构
        self.assertIsInstance(config, UnifiedResearchConfig)
        self.assertTrue(hasattr(config, "locale"))
        self.assertTrue(hasattr(config, "isolation_config"))

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_error_handling(self, mock_get_config):
        """测试错误处理"""
        # 模拟配置获取失败
        mock_get_config.side_effect = Exception("Config error")

        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        with self.assertRaises(Exception):
            IsolationConfigManager(self.mock_state, self.mock_config)


if __name__ == "__main__":
    unittest.main()
