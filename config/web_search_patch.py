
# Web Search 优化配置补丁
# 将此代码添加到你的搜索相关模块中

# 全局配置
WEB_SEARCH_OPTIMIZATION_CONFIG = {
  "description": "保守模式 - 保留更多内容，适合需要详细信息的场景",
  "search_config": {
    "max_search_results": 12,
    "enable_smart_filtering": true
  },
  "cleaning_config": {
    "min_quality_score": 0.2,
    "max_results": 15,
    "enable_keyword_extraction": true,
    "enable_enhanced_key_points": true,
    "filter_invalid_urls": true,
    "sort_by_quality": true
  },
  "content_limits": {
    "max_content_length": 4000,
    "max_raw_content_length": 6000
  }
}

def get_optimized_web_search_tool():
    """获取优化的 web search 工具"""
    from src.tools.search import get_web_search_tool
    
    search_config = WEB_SEARCH_OPTIMIZATION_CONFIG.get('search_config', {})
    return get_web_search_tool(
        max_search_results=search_config.get('max_search_results', 10),
        enable_smart_filtering=search_config.get('enable_smart_filtering', True)
    )

def get_optimized_search_filter():
    """获取优化的搜索过滤器"""
    from src.utils.tokens.content_processor import ContentProcessor
    from src.utils.common.search_result_filter import SearchResultFilter
    
    processor = ContentProcessor()
    cleaning_config = WEB_SEARCH_OPTIMIZATION_CONFIG.get('cleaning_config', {})
    return SearchResultFilter(processor, cleaning_config)
