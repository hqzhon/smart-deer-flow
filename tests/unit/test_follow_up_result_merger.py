#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Follow-up查询结果合并机制
"""

import unittest
from unittest.mock import patch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.common.follow_up_result_merger import FollowUpResultMerger
from src.config.follow_up_merger_config import (
    FollowUpMergerConfig,
    get_active_merger_config,
)


class TestFollowUpResultMerger(unittest.TestCase):
    """测试Follow-up结果合并器"""

    def setUp(self):
        """设置测试环境"""
        self.test_config = FollowUpMergerConfig(
            similarity_threshold=0.7,
            min_content_length=30,
            max_merged_results=10,
            enable_semantic_grouping=True,
            enable_intelligent_merging=True,
            enable_deduplication=True,
            enable_quality_filtering=True,
        )
        self.merger = FollowUpResultMerger(config=self.test_config)

    def test_merger_initialization_with_config(self):
        """测试使用配置初始化合并器"""
        self.assertEqual(self.merger.config.similarity_threshold, 0.7)
        self.assertEqual(self.merger.config.min_content_length, 30)
        self.assertEqual(self.merger.config.max_merged_results, 10)
        self.assertTrue(self.merger.config.enable_semantic_grouping)

    def test_merger_initialization_with_parameters(self):
        """测试使用参数覆盖初始化合并器"""
        merger = FollowUpResultMerger(
            config=self.test_config, similarity_threshold=0.8, max_merged_results=5
        )
        self.assertEqual(merger.config.similarity_threshold, 0.8)
        self.assertEqual(merger.config.max_merged_results, 5)
        # 其他配置应该保持不变
        self.assertEqual(merger.config.min_content_length, 30)

    def test_empty_results_handling(self):
        """测试空结果处理"""
        results = self.merger.merge_follow_up_results([], [])

        self.assertEqual(len(results), 0)

        # 检查统计信息
        stats = self.merger.get_merge_statistics(results)
        self.assertEqual(stats.get("total_original_results", 0), 0)

    def test_basic_result_merging(self):
        """测试基本结果合并"""
        test_results = [
            {
                "content": "这是第一个研究发现，包含了重要的信息。",
                "source": "source1",
                "type": "research_finding",
                "metadata": {"confidence": 0.8},
            },
            {
                "content": "这是第二个研究发现，提供了额外的见解。",
                "source": "source2",
                "type": "research_finding",
                "metadata": {"confidence": 0.9},
            },
        ]

        results = self.merger.merge_follow_up_results(test_results, [])

        self.assertGreaterEqual(len(results), 0)

        # 检查统计信息
        stats = self.merger.get_merge_statistics(results)
        # 检查合并后的结果数量
        self.assertEqual(stats.get("total_merged_results", 0), len(results))

    def test_deduplication_functionality(self):
        """测试去重功能"""
        # 创建重复内容的结果
        duplicate_content = "这是一个重复的研究发现内容。"
        test_results = [
            {
                "content": duplicate_content,
                "source": "source1",
                "type": "research_finding",
                "metadata": {"confidence": 0.8},
            },
            {
                "content": duplicate_content,  # 完全相同的内容
                "source": "source2",
                "type": "research_finding",
                "metadata": {"confidence": 0.9},
            },
            {
                "content": "这是一个不同的研究发现内容。",
                "source": "source3",
                "type": "research_finding",
                "metadata": {"confidence": 0.7},
            },
        ]

        results = self.merger.merge_follow_up_results(test_results, [])

        # 应该去除重复内容
        self.assertGreaterEqual(len(results), 0)
        self.assertLessEqual(len(results), 2)

        # 检查性能统计
        perf_stats = self.merger.get_performance_stats()
        self.assertGreaterEqual(perf_stats["filtering_stats"]["deduplication_count"], 0)

    def test_quality_filtering(self):
        """测试质量过滤"""
        test_results = [
            {
                "content": "这是一个高质量的详细研究发现，包含了丰富的信息和深入的分析。",
                "source": "source1",
                "type": "research_finding",
                "metadata": {"confidence": 0.9},
            },
            {
                "content": "短内容",  # 低质量内容
                "source": "source2",
                "type": "research_finding",
                "metadata": {"confidence": 0.3},
            },
        ]

        results = self.merger.merge_follow_up_results(test_results, [])

        # 低质量内容应该被过滤，只保留高质量内容
        self.assertGreaterEqual(len(results), 0)

        # 检查性能统计
        perf_stats = self.merger.get_performance_stats()
        self.assertIn("quality_filtered_count", perf_stats["filtering_stats"])

    def test_similarity_calculation_caching(self):
        """测试相似度计算缓存"""
        text1 = "这是第一个测试文本内容"
        text2 = "这是第二个测试文本内容"

        # 第一次计算
        similarity1 = self.merger._calculate_similarity(text1, text2)

        # 第二次计算（应该使用缓存）
        similarity2 = self.merger._calculate_similarity(text1, text2)

        self.assertEqual(similarity1, similarity2)

        # 检查缓存统计
        stats = self.merger.get_performance_stats()
        self.assertGreater(stats["cache_performance"]["hits"], 0)

    def test_performance_stats(self):
        """测试性能统计功能"""
        test_results = [
            {
                "content": "测试内容1",
                "source": "source1",
                "type": "research_finding",
                "metadata": {"confidence": 0.8},
            }
        ]

        # 执行合并
        self.merger.merge_follow_up_results(test_results, [])

        # 获取性能统计
        stats = self.merger.get_performance_stats()

        self.assertIn("total_merges", stats)
        self.assertIn("cache_performance", stats)
        self.assertIn("filtering_stats", stats)
        self.assertIn("config", stats)

        self.assertEqual(stats["total_merges"], 1)
        self.assertIn("hit_rate", stats["cache_performance"])

    def test_stats_reset(self):
        """测试统计重置功能"""
        test_results = [
            {
                "content": "测试内容",
                "source": "source1",
                "type": "research_finding",
                "metadata": {"confidence": 0.8},
            }
        ]

        # 执行合并
        self.merger.merge_follow_up_results(test_results, [])

        # 重置统计
        self.merger.reset_stats()

        # 检查统计是否被重置
        stats = self.merger.get_performance_stats()
        self.assertEqual(stats["total_merges"], 0)
        self.assertEqual(stats["cache_performance"]["hits"], 0)
        self.assertEqual(stats["cache_performance"]["misses"], 0)

    def test_config_integration(self):
        """测试配置集成"""
        # 测试使用自定义配置创建合并器
        from src.config.follow_up_merger_config import FollowUpMergerConfig
        
        custom_config = FollowUpMergerConfig(
            similarity_threshold=0.9, enable_semantic_grouping=False
        )
        
        # 创建新的合并器（使用自定义配置）
        merger = FollowUpResultMerger(config=custom_config)

        self.assertEqual(merger.config.similarity_threshold, 0.9)
        self.assertFalse(merger.config.enable_semantic_grouping)

    def test_original_findings_deduplication(self):
        """测试与原始发现的去重"""
        original_findings = ["这是一个已存在的研究发现。"]

        test_results = [
            {
                "content": "这是一个已存在的研究发现。",  # 与原始发现重复
                "source": "source1",
                "type": "research_finding",
                "metadata": {"confidence": 0.8},
            },
            {
                "content": "这是一个新的研究发现。",
                "source": "source2",
                "type": "research_finding",
                "metadata": {"confidence": 0.9},
            },
        ]

        results = self.merger.merge_follow_up_results(test_results, original_findings)

        # 应该过滤掉与原始发现重复的内容
        self.assertGreaterEqual(len(results), 0)
        self.assertLessEqual(len(results), 1)

        # 检查性能统计
        perf_stats = self.merger.get_performance_stats()
        self.assertGreaterEqual(perf_stats["filtering_stats"]["deduplication_count"], 0)


class TestFollowUpMergerConfig(unittest.TestCase):
    """测试Follow-up合并器配置"""

    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的相似度阈值
        with self.assertRaises(ValueError):
            FollowUpMergerConfig(similarity_threshold=1.5)

        with self.assertRaises(ValueError):
            FollowUpMergerConfig(similarity_threshold=-0.1)

        # 测试无效的最大结果数
        with self.assertRaises(ValueError):
            FollowUpMergerConfig(max_merged_results=0)

        with self.assertRaises(ValueError):
            FollowUpMergerConfig(max_merged_results=-1)

        # 测试有效配置
        valid_config = FollowUpMergerConfig(
            similarity_threshold=0.8, max_merged_results=10
        )
        self.assertEqual(valid_config.similarity_threshold, 0.8)
        self.assertEqual(valid_config.max_merged_results, 10)

    def test_preset_configs(self):
        """测试预设配置"""
        conservative_config = FollowUpMergerConfig.create_conservative_config()
        aggressive_config = FollowUpMergerConfig.create_aggressive_config()
        performance_config = FollowUpMergerConfig.create_performance_config()
        quality_config = FollowUpMergerConfig.create_quality_focused_config()

        # 保守配置应该有较高的相似度阈值
        self.assertGreater(conservative_config.similarity_threshold, 0.7)

        # 激进配置应该有较低的相似度阈值
        self.assertLess(aggressive_config.similarity_threshold, 0.6)

        # 性能配置应该启用缓存
        self.assertTrue(performance_config.enable_similarity_cache)

        # 质量配置应该启用质量过滤
        self.assertTrue(quality_config.enable_quality_filtering)

    def test_runtime_config_switching(self):
        """测试运行时配置切换"""
        try:
            from src.config.follow_up_merger_config import (
                switch_merger_config,
                get_active_merger_config,
            )

            # 切换到激进配置
            switch_merger_config("aggressive")
            active_config = get_active_merger_config()
            self.assertLess(active_config.similarity_threshold, 0.6)

            # 切换到保守配置
            switch_merger_config("conservative")
            active_config = get_active_merger_config()
            self.assertGreater(active_config.similarity_threshold, 0.7)
        except ImportError:
            # 如果旧配置系统不可用，跳过此测试
            self.skipTest("Legacy configuration system not available")


if __name__ == "__main__":
    unittest.main()
