#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索结果过滤器单元测试
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.common.search_result_filter import SearchResultCleaner, SearchResultFilter
from src.utils.tokens.content_processor import ContentProcessor


class TestSearchResultCleaner(unittest.TestCase):
    """搜索结果清洗器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.cleaner = SearchResultCleaner()
    
    def test_encoding_fixes(self):
        """测试编码修复功能"""
        test_cases = [
            {
                'name': 'Unicode替换字符',
                'input': '这是一个测试�文本�包含替换字符',
                'should_fix': True
            },
            {
                'name': 'HTML实体编码',
                'input': '这是&lt;测试&gt;内容&amp;示例&quot;文本&quot;',
                'should_fix': True
            },
            {
                'name': 'URL编码',
                'input': '这是%E6%B5%8B%E8%AF%95%20%E5%86%85%E5%AE%B9',
                'should_fix': True
            },
            {
                'name': '连续问号乱码',
                'input': '正常文本???乱码???更多内容',
                'should_fix': True
            },
            {
                'name': '方块字符乱码',
                'input': '正常文本■■■乱码■更多内容',
                'should_fix': True
            },
            {
                'name': '双重编码问题',
                'input': '&amp;lt;script&amp;gt;alert(&amp;quot;test&amp;quot;)&amp;lt;/script&amp;gt;',
                'should_fix': True
            },
            {
                'name': '控制字符和特殊空白字符',
                'input': '正常文本\x00\x01\x02\u2000\u2001\u2002更多内容',
                'should_fix': True
            },
            {
                'name': '混合乱码',
                'input': '这是测试�内容%20包含&amp;多种■乱码???问题',
                'should_fix': True
            },
            {
                'name': '正常中英文混合内容',
                'input': '这是正常的中英文混合内容 This is normal mixed content',
                'should_fix': False
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['name']):
                original = case['input']
                fixed = self.cleaner._fix_encoding_issues(original)
                
                if case['should_fix']:
                    # 应该被修复的情况，检查是否有改善
                    # 检查修复后的文本不应包含明显的乱码
                    self.assertNotIn('�', fixed, "不应包含Unicode替换字符")
                    self.assertNotIn('■', fixed, "不应包含方块字符")
                    self.assertNotIn('%20', fixed, "不应包含URL编码")
                    # 对于连续问号，检查是否有所减少
                    if '???' in original:
                        original_question_marks = original.count('?')
                        fixed_question_marks = fixed.count('?')
                        self.assertLessEqual(fixed_question_marks, original_question_marks, 
                                           "连续问号应该被减少或移除")
                else:
                    # 正常内容不应被过度修改
                    self.assertEqual(len(original), len(fixed), f"正常内容不应被大幅修改: {case['name']}")
    
    def test_noise_pattern_removal(self):
        """测试噪音模式移除功能"""
        test_cases = [
            {
                'name': '广告和推广内容',
                'input': 'Click here to read more! Subscribe to our newsletter! This is useful content.',
                'expected_keywords': ['useful', 'content']
            },
            {
                'name': '联系信息',
                'input': 'Contact us at info@example.com or call 123-456-7890. Learn Python programming.',
                'expected_keywords': ['Learn', 'Python', 'programming']
            },
            {
                'name': 'JavaScript代码',
                'input': 'alert("popup"); console.log("debug"); This is useful content about programming.',
                'expected_keywords': ['useful', 'content', 'programming']
            },
            {
                'name': 'HTML标签',
                'input': '<script>alert("hello");</script>Python is a great language for beginners.',
                'expected_keywords': ['Python', 'great', 'language', 'beginners']
            },
            {
                'name': 'CSS样式',
                'input': '<style>body{color:red;}</style>This article covers advanced topics.',
                'expected_keywords': ['article', 'covers', 'advanced', 'topics']
            },
            {
                'name': 'API响应乱码',
                'input': 'Error: Failed to process request due to invalid parameters',
                'expected_keywords': ['Error', 'Failed', 'process', 'request', 'invalid', 'parameters']
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['name']):
                cleaned = self.cleaner.clean_text(case['input'])
                
                # 检查清洗后的文本不为空
                self.assertGreater(len(cleaned.strip()), 0, f"清洗后文本不应为空: {case['name']}")
                
                # 检查是否保留了有用的关键词
                for keyword in case['expected_keywords']:
                    self.assertIn(keyword.lower(), cleaned.lower(), 
                                f"应该保留关键词 '{keyword}' 在案例 '{case['name']}' 中")
    
    def test_content_quality_assessment(self):
        """测试内容质量评估功能"""
        test_cases = [
            {
                'text': 'Python is a high-level programming language. It\'s known for its simplicity and readability.',
                'expected_range': (0.7, 1.0),
                'description': '高质量技术内容'
            },
            {
                'text': 'Click here! Buy now! Limited offer!',
                'expected_range': (0.0, 0.3),
                'description': '低质量广告内容'
            },
            {
                'text': 'Short.',
                'expected_range': (0.0, 0.2),
                'description': '过短内容'
            },
            {
                'text': 'This comprehensive guide covers machine learning algorithms from basic concepts to advanced implementations.',
                'expected_range': (0.5, 1.0),
                'description': '中高质量教育内容'
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case['description']):
                score = self.cleaner.calculate_content_quality(case['text'])
                min_score, max_score = case['expected_range']
                
                self.assertGreaterEqual(score, min_score, 
                                      f"质量分数应该 >= {min_score}: {case['description']}")
                self.assertLessEqual(score, max_score, 
                                   f"质量分数应该 <= {max_score}: {case['description']}")
    
    def test_performance_requirements(self):
        """测试性能要求"""
        import time
        
        # 创建大文本用于性能测试
        large_text = "This is a sample text with noise patterns. " * 1000
        
        # 测试清洗性能
        start_time = time.time()
        for _ in range(10):
            self.cleaner.clean_text(large_text)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        self.assertLess(avg_time, 1.0, "单次清洗应在1秒内完成")
        
        # 测试质量评估性能
        start_time = time.time()
        for _ in range(10):
            self.cleaner.calculate_content_quality(large_text)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        self.assertLess(avg_time, 0.5, "单次质量评估应在0.5秒内完成")


class TestSearchResultFilter(unittest.TestCase):
    """搜索结果过滤器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟的ContentProcessor
        class MockContentProcessor:
            def sanitize_content(self, content):
                return content
        
        self.filter = SearchResultFilter(MockContentProcessor())
    
    def test_integrated_filtering(self):
        """测试集成过滤功能"""
        # 模拟搜索结果
        search_results = [
            {
                'title': 'Python Programming Guide',
                'content': 'This is a comprehensive guide to Python programming. Learn advanced techniques and best practices.',
                'url': 'https://realpython.com/python-guide'
            },
            {
                'title': 'Buy Python Course Now!',
                'content': 'Limited time offer! Subscribe to our newsletter! Click here!',
                'url': 'https://spam-site.com/course'
            },
            {
                'title': 'Machine Learning with Python',
                'content': 'Machine learning enables computers to learn from data. Python provides excellent libraries like scikit-learn.',
                'url': 'https://github.com/ml-guide'
            }
        ]
        
        # 使用严格配置处理
        config = {
            'min_quality_score': 0.5,
            'max_results': 10,
            'enable_keyword_extraction': True,
            'filter_invalid_urls': True,
            'sort_by_quality': True
        }
        
        self.filter.cleaning_config = config
        processed_results = self.filter._process_search_results_with_config(search_results)
        
        # 检查过滤效果
        self.assertGreater(len(processed_results), 0, "应该有处理后的结果")
        self.assertLess(len(processed_results), len(search_results), "应该过滤掉低质量结果")
        
        # 检查结果质量
        for result in processed_results:
            self.assertIn('quality_score', result, "结果应包含质量分数")
            self.assertGreaterEqual(result['quality_score'], config['min_quality_score'], 
                                  "结果质量分数应满足最低要求")


if __name__ == '__main__':
    unittest.main()