#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置系统测试

测试新的统一配置系统是否正确加载和使用反射和Follow-up合并器配置。
"""

import unittest
import tempfile
import os


class TestUnifiedConfigSystem(unittest.TestCase):
    """测试统一配置系统"""

    def setUp(self):
        """设置测试环境"""
        self.test_config_content = """
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "test-key"

reflection:
  reflection_enabled: true
  reflection_max_loops: 5
  reflection_sufficiency_threshold: 0.8
  reflection_knowledge_gap_threshold: 0.7
  reflection_max_followup_queries: 3
  reflection_skip_initial_stage: false
  reflection_temperature: 0.7

followup_merger:
  similarity_threshold: 0.75
  min_content_length: 50
  max_merged_results: 15
  enable_semantic_grouping: true
  enable_intelligent_merging: true
  enable_deduplication: true
  enable_quality_filtering: true
  quality_threshold: 0.6
  confidence_weight: 0.4
  relevance_weight: 0.3
  content_quality_weight: 0.3
  max_sentences_per_result: 5
  max_key_points: 8
  preserve_source_info: true
  enable_similarity_cache: true
  max_cache_size: 1000
  active_config_preset: "default"
  enable_config_switching: true
  enable_detailed_logging: false
  log_merge_statistics: true
"""

    def test_config_loading(self):
        """测试配置加载"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(self.test_config_content)
            config_path = f.name

        try:
            from src.config.models import AppSettings
            import yaml

            # 直接加载配置文件
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            app_settings = AppSettings(**config)

            # 验证反射配置（简化版）
            reflection_config = app_settings.reflection
            self.assertTrue(reflection_config.enabled)
            self.assertEqual(reflection_config.max_loops, 2)  # 使用默认值
            self.assertEqual(reflection_config.quality_threshold, 0.75)  # 使用默认值

            # 验证Follow-up合并器配置
            merger_config = app_settings.followup_merger
            self.assertEqual(
                merger_config.similarity_threshold, 0.75
            )  # 从配置文件加载的值
            self.assertEqual(merger_config.min_content_length, 50)
            self.assertEqual(merger_config.max_merged_results, 15)
            self.assertTrue(merger_config.enable_semantic_grouping)
            self.assertTrue(merger_config.enable_intelligent_merging)
            self.assertTrue(merger_config.enable_deduplication)
            self.assertTrue(merger_config.enable_quality_filtering)
            self.assertEqual(merger_config.quality_threshold, 0.6)
            self.assertEqual(merger_config.confidence_weight, 0.4)
            self.assertEqual(merger_config.relevance_weight, 0.3)
            self.assertEqual(merger_config.content_quality_weight, 0.3)
            self.assertEqual(merger_config.max_sentences_per_result, 5)
            self.assertEqual(merger_config.max_key_points, 8)
            self.assertTrue(merger_config.preserve_source_info)
            self.assertTrue(merger_config.enable_similarity_cache)
            self.assertEqual(merger_config.max_cache_size, 1000)
            self.assertEqual(merger_config.active_config_preset, "default")
            self.assertTrue(merger_config.enable_config_switching)
            self.assertFalse(merger_config.enable_detailed_logging)
            self.assertTrue(merger_config.log_merge_statistics)

        finally:
            # 清理临时文件
            os.unlink(config_path)

    def test_reflection_manager_integration(self):
        """测试反射管理器与统一配置系统的集成"""
        try:
            from src.utils.reflection.reflection_manager import ReflectionManager

            # 创建反射管理器（应该自动加载配置）
            manager = ReflectionManager()

            # 验证配置是否正确加载
            self.assertIsNotNone(manager.config)

        except ImportError as e:
            self.skipTest(f"Reflection manager not available: {e}")

    def test_follow_up_merger_integration(self):
        """测试Follow-up合并器与统一配置系统的集成"""
        try:
            from src.utils.common.follow_up_result_merger import FollowUpResultMerger

            # 创建合并器（应该自动加载配置）
            merger = FollowUpResultMerger()

            # 验证配置是否正确加载
            self.assertIsNotNone(merger.config)

        except ImportError as e:
            self.skipTest(f"Follow-up merger not available: {e}")

    def test_reflection_integration_config(self):
        """测试反射集成器与统一配置系统的集成"""
        try:
            from src.utils.reflection.reflection_integration import ReflectionIntegrator

            # 创建反射集成器（应该自动加载配置）
            integrator = ReflectionIntegrator()

            # 验证配置是否正确加载
            self.assertIsNotNone(integrator.config)
            self.assertIsInstance(integrator.config, dict)

        except ImportError as e:
            self.skipTest(f"Reflection integrator not available: {e}")

    def test_default_config_fallback(self):
        """测试默认配置回退机制"""
        from src.config.models import ReflectionSettings, FollowUpMergerSettings

        # 测试反射设置默认值（简化版）
        reflection_settings = ReflectionSettings()
        self.assertTrue(reflection_settings.enabled)
        self.assertEqual(reflection_settings.max_loops, 2)
        self.assertEqual(reflection_settings.quality_threshold, 0.75)

        # 测试Follow-up合并器设置默认值
        merger_settings = FollowUpMergerSettings()
        self.assertEqual(merger_settings.similarity_threshold, 0.7)
        self.assertEqual(merger_settings.min_content_length, 50)  # 正确的默认值
        self.assertEqual(merger_settings.max_merged_results, 20)  # 正确的默认值
        self.assertTrue(merger_settings.enable_semantic_grouping)
        self.assertTrue(merger_settings.enable_intelligent_merging)


if __name__ == "__main__":
    unittest.main()
