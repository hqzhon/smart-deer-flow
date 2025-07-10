#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索结果清洗配置文件

这个文件包含了各种清洗配置选项，用户可以根据具体需求进行调整。
"""

from typing import Dict, Any, List, Set


class CleaningConfig:
    """清洗配置类"""
    
    # 基础配置
    BASIC_CONFIG = {
        'min_quality_score': 0.3,
        'max_results': 20,
        'enable_keyword_extraction': True,
        'enable_enhanced_key_points': True,
        'filter_invalid_urls': True,
        'sort_by_quality': True,
    }
    
    # 严格配置（高质量要求）
    STRICT_CONFIG = {
        'min_quality_score': 0.6,
        'max_results': 10,
        'enable_keyword_extraction': True,
        'enable_enhanced_key_points': True,
        'filter_invalid_urls': True,
        'sort_by_quality': True,
    }
    
    # 宽松配置（保留更多结果）
    LENIENT_CONFIG = {
        'min_quality_score': 0.1,
        'max_results': 50,
        'enable_keyword_extraction': True,
        'enable_enhanced_key_points': True,
        'filter_invalid_urls': False,
        'sort_by_quality': True,
    }
    
    # 快速配置（性能优先）
    FAST_CONFIG = {
        'min_quality_score': 0.2,
        'max_results': 15,
        'enable_keyword_extraction': False,
        'enable_enhanced_key_points': False,
        'filter_invalid_urls': True,
        'sort_by_quality': False,
    }


class NoisePatterns:
    """噪音模式配置"""
    
    # 基础噪音模式（改进版，匹配完整短语）
    BASIC_PATTERNS = [
        r'\b(?:click here|read more|learn more|see more|view all)\b[^.!?]*[.!?]?',
        r'\b(?:advertisement|sponsored|ads?)\b[^.!?]*[.!?]?',
        r'\b(?:cookie|privacy policy|terms of service)\b[^.!?]*[.!?]?',
        r'\b(?:subscribe|newsletter|sign up)\b[^.!?]*[.!?]?',
        r'\b(?:share|like|follow|tweet)\b[^.!?]*[.!?]?',
        r'\b(?:loading|please wait|error)\b[^.!?]*[.!?]?',
        r'\b(?:javascript|enable js|browser)\b[^.!?]*[.!?]?',
        # 联系信息模式
        r'\b(?:contact us|call us|email us)\b[^.!?]*[.!?]?',
        r'\b(?:at|or call)\s+[\w@.-]+\s*[.!?]?',
        # JavaScript和脚本模式
        r'\balert\s*\([^)]*\)\s*;?',
        r'\bconsole\.[a-z]+\s*\([^)]*\)\s*;?',
    ]
    
    # 扩展噪音模式（包含更多模式）
    EXTENDED_PATTERNS = BASIC_PATTERNS + [
        r'\b(?:buy now|order now|purchase|sale|discount)\b',
        r'\b(?:free trial|limited time|exclusive offer)\b',
        r'\b(?:contact us|call now|email us)\b',
        r'\b(?:download|install|upgrade)\b',
        r'\b(?:register|login|account|profile)\b',
        r'\b(?:menu|navigation|sidebar|footer)\b',
        r'\b(?:home|about|services|products)\b',
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}',  # 时间戳
        r'\b\w+@\w+\.\w+\b',  # 邮箱
        r'\b\d{3}-\d{3}-\d{4}\b',  # 电话号码
        r'\b(?:copyright|©|all rights reserved)\b',
    ]
    
    # 中文噪音模式
    CHINESE_PATTERNS = [
        r'\b(?:点击这里|阅读更多|了解更多|查看更多|查看全部)\b',
        r'\b(?:广告|赞助|推广)\b',
        r'\b(?:隐私政策|服务条款|用户协议)\b',
        r'\b(?:订阅|注册|登录|关注)\b',
        r'\b(?:分享|点赞|转发|收藏)\b',
        r'\b(?:加载中|请稍候|错误)\b',
        r'\b(?:联系我们|客服|咨询)\b',
        r'\b(?:立即购买|马上下单|限时优惠)\b',
    ]


class StopWords:
    """停用词配置"""
    
    # 英文停用词
    ENGLISH_STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom',
        'if', 'then', 'else', 'than', 'as', 'so', 'too', 'very', 'just', 'now', 'here', 'there',
        'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
    }
    
    # 中文停用词
    CHINESE_STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
        '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '就是',
        '他', '时候', '过', '下', '可以', '出', '比', '还', '多', '个', '她', '真的', '等', '什么',
        '这个', '我们', '来', '对', '生活', '学习', '工作', '问题', '系统', '社会', '设计', '方法',
        '但是', '因为', '所以', '如果', '虽然', '然而', '而且', '或者', '以及', '关于', '根据',
        '通过', '由于', '为了', '除了', '包括', '特别', '尤其', '主要', '重要', '基本', '一般'
    }
    
    # 技术相关停用词
    TECH_STOP_WORDS = {
        'code', 'example', 'sample', 'demo', 'test', 'function', 'method', 'class', 'variable',
        'parameter', 'argument', 'return', 'value', 'type', 'string', 'number', 'boolean',
        'array', 'object', 'null', 'undefined', 'true', 'false', 'import', 'export',
        'module', 'package', 'library', 'framework', 'api', 'url', 'http', 'https'
    }


class DomainFilters:
    """域名过滤配置"""
    
    # 社交媒体域名
    SOCIAL_MEDIA_DOMAINS = {
        'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
        'pinterest.com', 'youtube.com', 'tiktok.com', 'reddit.com',
        'snapchat.com', 'discord.com', 'telegram.org', 'whatsapp.com'
    }
    
    # 广告和营销域名
    AD_DOMAINS = {
        'doubleclick.net', 'googleadservices.com', 'googlesyndication.com',
        'amazon-adsystem.com', 'adsystem.amazon.com', 'googletagmanager.com'
    }
    
    # 低质量内容域名
    LOW_QUALITY_DOMAINS = {
        'spam.com', 'clickbait.com', 'fake-news.com',
        'content-farm.com', 'auto-generated.com'
    }
    
    # 可信域名（白名单）
    TRUSTED_DOMAINS = {
        'stackoverflow.com', 'github.com', 'medium.com', 'dev.to',
        'realpython.com', 'python.org', 'docs.python.org',
        'mozilla.org', 'w3schools.com', 'freecodecamp.org',
        'coursera.org', 'edx.org', 'udemy.com', 'khan academy.org'
    }


class QualityMetrics:
    """质量评估指标配置"""
    
    # 内容长度权重
    LENGTH_WEIGHTS = {
        'very_short': (0, 50, 0.1),      # 0-50字符，权重0.1
        'short': (50, 200, 0.3),         # 50-200字符，权重0.3
        'medium': (200, 1000, 0.5),      # 200-1000字符，权重0.5
        'long': (1000, 2000, 0.3),       # 1000-2000字符，权重0.3
        'very_long': (2000, float('inf'), 0.1)  # 2000+字符，权重0.1
    }
    
    # 句子完整性权重
    SENTENCE_WEIGHTS = {
        'no_sentences': 0.0,
        'one_sentence': 0.2,
        'few_sentences': 0.4,
        'many_sentences': 0.6
    }
    
    # 信息密度权重
    DENSITY_WEIGHTS = {
        'low_density': 0.1,
        'medium_density': 0.3,
        'high_density': 0.5
    }


def get_config_by_use_case(use_case: str) -> Dict[str, Any]:
    """根据使用场景获取配置
    
    Args:
        use_case: 使用场景 ('basic', 'strict', 'lenient', 'fast')
        
    Returns:
        Dict[str, Any]: 对应的配置字典
    """
    config_map = {
        'basic': CleaningConfig.BASIC_CONFIG,
        'strict': CleaningConfig.STRICT_CONFIG,
        'lenient': CleaningConfig.LENIENT_CONFIG,
        'fast': CleaningConfig.FAST_CONFIG
    }
    
    return config_map.get(use_case, CleaningConfig.BASIC_CONFIG)


def get_noise_patterns_by_language(language: str) -> List[str]:
    """根据语言获取噪音模式
    
    Args:
        language: 语言 ('en', 'zh', 'mixed')
        
    Returns:
        List[str]: 噪音模式列表
    """
    if language == 'en':
        return NoisePatterns.EXTENDED_PATTERNS
    elif language == 'zh':
        return NoisePatterns.BASIC_PATTERNS + NoisePatterns.CHINESE_PATTERNS
    elif language == 'mixed':
        return NoisePatterns.EXTENDED_PATTERNS + NoisePatterns.CHINESE_PATTERNS
    else:
        return NoisePatterns.BASIC_PATTERNS


def get_stop_words_by_domain(domain: str) -> Set[str]:
    """根据领域获取停用词
    
    Args:
        domain: 领域 ('general', 'tech', 'chinese')
        
    Returns:
        Set[str]: 停用词集合
    """
    if domain == 'tech':
        return StopWords.ENGLISH_STOP_WORDS | StopWords.TECH_STOP_WORDS
    elif domain == 'chinese':
        return StopWords.CHINESE_STOP_WORDS
    elif domain == 'mixed':
        return StopWords.ENGLISH_STOP_WORDS | StopWords.CHINESE_STOP_WORDS
    else:
        return StopWords.ENGLISH_STOP_WORDS


# 预定义配置组合
PREDEFINED_CONFIGS = {
    'academic': {
        'config': CleaningConfig.STRICT_CONFIG,
        'noise_patterns': NoisePatterns.EXTENDED_PATTERNS,
        'stop_words': StopWords.ENGLISH_STOP_WORDS,
        'invalid_domains': DomainFilters.SOCIAL_MEDIA_DOMAINS | DomainFilters.AD_DOMAINS
    },
    'news': {
        'config': CleaningConfig.BASIC_CONFIG,
        'noise_patterns': NoisePatterns.BASIC_PATTERNS,
        'stop_words': StopWords.ENGLISH_STOP_WORDS,
        'invalid_domains': DomainFilters.AD_DOMAINS
    },
    'tech': {
        'config': CleaningConfig.BASIC_CONFIG,
        'noise_patterns': NoisePatterns.EXTENDED_PATTERNS,
        'stop_words': StopWords.ENGLISH_STOP_WORDS | StopWords.TECH_STOP_WORDS,
        'invalid_domains': DomainFilters.SOCIAL_MEDIA_DOMAINS
    },
    'chinese': {
        'config': CleaningConfig.BASIC_CONFIG,
        'noise_patterns': NoisePatterns.BASIC_PATTERNS + NoisePatterns.CHINESE_PATTERNS,
        'stop_words': StopWords.CHINESE_STOP_WORDS,
        'invalid_domains': DomainFilters.SOCIAL_MEDIA_DOMAINS
    }
}