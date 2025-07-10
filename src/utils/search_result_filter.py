# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.utils.content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class SearchResultCleaner:
    """高效的本地搜索结果清洗器"""
    
    def __init__(self):
        # 改进后的噪音模式（匹配完整短语）
        self.noise_patterns = [
            r'\b(?:click here|read more|learn more|see more|view all)\b[^.!?]*[.!?]?',
            r'\b(?:advertisement|sponsored|ads?)\b[^.!?]*[.!?]?',
            r'\b(?:cookie|privacy policy|terms of service)\b[^.!?]*[.!?]?',
            r'\b(?:subscribe|newsletter|sign up)\b[^.!?]*[.!?]?',
            r'\b(?:share|like|follow|tweet)\b[^.!?]*[.!?]?',
            r'\b(?:loading|please wait)\b[^.!?]*[.!?]?',
            r'\berror\s*:?\s*(?:loading|occurred|404|500|not found)\b[^.!?]*[.!?]?',  # 只匹配明确的错误消息
            r'\b(?:javascript|enable js|browser)\b[^.!?]*[.!?]?',
            # 联系信息模式
            r'\b(?:contact us|call us|email us)\b[^.!?]*[.!?]?',
            r'\b(?:at|or call)\s+[\w@.-]+\s*[.!?]?',
            # 时间戳、邮箱、电话号码
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}',
            r'\b\w+@\w+\.\w+\b',
            r'\b\d{3}-\d{3}-\d{4}\b',
        ]
        
        # 编译正则表达式以提高性能
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.noise_patterns]
        
        # 停用词列表（简化版）
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # 无效域名模式
        self.invalid_domains = {
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'pinterest.com', 'youtube.com', 'tiktok.com', 'reddit.com'
        }
    
    def clean_text(self, text: str) -> str:
        """清洗文本内容"""
        if not text or not isinstance(text, str):
            return ""
        
        # 0. 移除脚本和样式内容
        text = self._remove_scripts_and_styles(text)
        
        # 1. 基础清理
        text = self._basic_clean(text)
        
        # 2. 移除噪音模式
        text = self._remove_noise_patterns(text)
        
        # 3. 标准化空白字符
        text = self._normalize_whitespace(text)
        
        # 4. 移除过短或过长的内容
        if len(text.strip()) < 10 or len(text) > 5000:
            return ""
        
        return text.strip()
    
    def _basic_clean(self, text: str) -> str:
        """基础文本清理"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 处理乱码和编码问题
        text = self._fix_encoding_issues(text)
        
        # 移除特殊字符和控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # 移除多余的标点符号
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # 移除URL
        text = re.sub(r'https?://\S+', '', text)
        
        return text
    
    def _remove_noise_patterns(self, text: str) -> str:
        """移除噪音模式"""
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        return self._post_clean_noise(text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除行首行尾空白
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # 移除多余的换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _post_clean_noise(self, text: str) -> str:
        """噪音清理后的文本后处理"""
        # 移除多余的连接词和标点
        text = re.sub(r'\b(?:at|or|call|to|and|the|a|an)\s+[.!?]', '', text)
        # 清理多余的空格和标点
        text = re.sub(r'\s+[.!?]\s+', ' ', text)
        # 移除孤立的标点符号
        text = re.sub(r'\s+[.!?]$', '', text)
        text = re.sub(r'^[.!?]\s+', '', text)
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_scripts_and_styles(self, text: str) -> str:
        """移除脚本和样式内容"""
        # 移除script标签及其内容
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除style标签及其内容
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除JavaScript代码片段
        text = re.sub(r'\balert\s*\([^)]*\)\s*;?', '', text)
        text = re.sub(r'\bconsole\.[a-z]+\s*\([^)]*\)\s*;?', '', text)
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """修复编码问题和乱码"""
        if not text:
            return text
        
        try:
            # 1. 处理Unicode替换字符（明确的乱码标记）
            if '\ufffd' in text:
                text = text.replace('\ufffd', '')
            
            # 2. 处理HTML实体编码（仅处理常见的HTML实体）
            import html
            # 只处理明确的HTML实体，避免误处理正常文本
            if '&' in text and (';' in text):
                # 检查是否包含常见HTML实体
                if any(entity in text for entity in ['&lt;', '&gt;', '&amp;', '&quot;', '&#']):
                    text = html.unescape(text)
            
            # 3. 处理URL编码（更保守的方法）
            import urllib.parse
            # 只有当文本看起来像URL编码时才处理
            if '%' in text:
                # 检查是否有URL编码模式
                url_encoded_pattern = r'%[0-9A-Fa-f]{2}'
                url_matches = re.findall(url_encoded_pattern, text)
                if len(url_matches) >= 1:  # 至少1个编码字符
                    try:
                        decoded = urllib.parse.unquote(text, errors='ignore')
                        # 检查解码是否有意义：
                        # 1. 解码后不能为空
                        # 2. 解码后应该包含可读字符
                        # 3. URL编码字符占比较高时优先解码，或者解码后明显更可读
                        url_encoded_ratio = len(''.join(url_matches)) / len(text)
                        if (decoded and decoded.strip() and 
                            (len(decoded) > len(text) * 0.5 or url_encoded_ratio > 0.15 or 
                             (decoded != text and len(decoded.strip()) > len(text.strip()) * 0.8))):
                            text = decoded
                    except Exception:
                        pass
            
            # 4. 移除明确的乱码模式（更保守）
            # 只移除连续4个或以上的问号（明确的乱码）
            text = re.sub(r'\?{4,}', '', text)
            
            # 移除方块字符（单个或连续的都是乱码）
            text = re.sub(r'[\u25a0\u25a1\u2588■▪▫]+', '', text)
            
            # 移除零宽字符和其他不可见字符
            text = re.sub(r'[\ufeff\u200b-\u200f\u2028\u2029]', '', text)
            
            # 5. 处理双重编码问题（更谨慎）
            # 只有当检测到明确的双重编码模式时才处理
            double_encoding_indicators = ['Ã¡', 'Ã©', 'Ã­', 'Ã³', 'Ãº', 'Ã±', 'Ã§']
            if any(indicator in text for indicator in double_encoding_indicators):
                # 计算这些指示符的密度
                indicator_count = sum(text.count(indicator) for indicator in double_encoding_indicators)
                if indicator_count >= 2:  # 至少出现2次才认为是双重编码
                    try:
                        text_bytes = text.encode('latin-1')
                        decoded = text_bytes.decode('utf-8', errors='ignore')
                        # 只有解码后看起来更正常时才使用
                        if len(decoded) >= len(text) * 0.7:  # 解码后不能丢失太多内容
                            text = decoded
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
            
            # 6. 移除控制字符（保留基本空白字符）
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            
            # 7. 标准化特殊空白字符为普通空格
            text = re.sub(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]', ' ', text)
            
            # 8. 清理多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            # 如果处理过程中出现任何错误，返回原始文本
            logger.warning(f"Error fixing encoding issues: {e}")
            return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取关键词"""
        if not text:
            return []
        
        # 转换为小写并分词
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # 过滤停用词
        keywords = [word for word in words if word not in self.stop_words]
        
        # 统计词频
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        if not url or not isinstance(url, str):
            return False
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # 检查是否为无效域名
            domain = parsed.netloc.lower()
            for invalid_domain in self.invalid_domains:
                if invalid_domain in domain:
                    return False
            
            return True
        except Exception:
            return False
    
    def calculate_content_quality(self, content: str) -> float:
        """计算内容质量分数 (0-1)"""
        if not content:
            return 0.0
        
        score = 0.0
        
        # 长度分数 (适中长度得分更高)
        length = len(content)
        if 50 <= length <= 1000:
            score += 0.3
        elif 1000 < length <= 2000:
            score += 0.2
        elif length > 2000:
            score += 0.1
        
        # 句子完整性分数
        sentences = content.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        if complete_sentences >= 2:
            score += 0.3
        
        # 信息密度分数 (关键词密度)
        keywords = self.extract_keywords(content, 5)
        if len(keywords) >= 3:
            score += 0.2
        
        # 可读性分数 (基于标点符号和结构)
        if '.' in content and (',' in content or ';' in content):
            score += 0.2
        
        return min(score, 1.0)


class SearchResultFilter:
    """Search result filter for content processing and formatting"""

    # Smart filtering threshold configuration
    SMART_FILTERING_THRESHOLD_RATIO = 0.6  # 60% threshold
    DEFAULT_BATCH_SIZE = 3
    MAX_SECONDARY_RESULTS = 5
    
    # 清洗配置选项
    DEFAULT_CLEANING_CONFIG = {
        'min_quality_score': 0.3,  # 最低质量分数
        'max_results': 20,  # 最大结果数
        'enable_keyword_extraction': True,  # 启用关键词提取
        'enable_enhanced_key_points': True,  # 启用增强关键点提取
        'filter_invalid_urls': True,  # 过滤无效URL
        'sort_by_quality': True,  # 按质量排序
    }

    def __init__(self, content_processor: ContentProcessor, cleaning_config: Optional[Dict[str, Any]] = None):
        """Initialize search result filter

        Args:
            content_processor: Content processor instance
            cleaning_config: 清洗配置选项
        """
        self.content_processor = content_processor
        self.cleaner = SearchResultCleaner()  # 初始化清洗器
        self.cleaning_config = {**self.DEFAULT_CLEANING_CONFIG, **(cleaning_config or {})}
        # 添加新的配置选项
        self.cleaning_config.setdefault('filter_raw_content', True)  # 是否过滤raw_content
        self.cleaning_config.setdefault('prefer_content_over_raw', True)  # 优先使用content而非raw_content



    def should_enable_smart_filtering(
        self, search_results: List[Dict[str, Any]], model_name: str
    ) -> bool:
        """Determine whether smart filtering should be enabled

        Args:
            search_results: List of search results
            model_name: Model name

        Returns:
            bool: Whether smart filtering should be enabled
        """
        if not search_results:
            return False

        # Always return True to enable smart filtering when there are search results
        # The actual enable/disable control is handled by the configuration
        return True

    def get_smart_filtering_threshold(self, model_name: str) -> int:
        """Get smart filtering threshold

        Args:
            model_name: Model name

        Returns:
            int: Smart filtering threshold (token count)
        """
        limits = self.content_processor.get_model_limits(model_name)
        return int(limits.input_limit * self.SMART_FILTERING_THRESHOLD_RATIO)

    def filter_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        model_name: str,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process and format search results without LLM filtering

        Args:
            query: User query
            search_results: Original search results
            model_name: Model name
            max_results: Maximum number of results

        Returns:
            Dictionary of processed results
        """
        if not search_results:
            return self._create_empty_result()

        # Limit search results count
        if max_results:
            search_results = search_results[:max_results]

        logger.info(f"Processing {len(search_results)} search results")

        # Process and clean search results
        processed_results = self._process_search_results(search_results)

        return {
            "filtered_results": processed_results,
            "summary": f"Processed {len(processed_results)} search results",
            "total_filtered": len(processed_results),
            "total_original": len(search_results),
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "filtered_results": [],
            "summary": "No relevant search results found",
            "total_filtered": 0,
            "total_original": 0,
        }

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results as text"""
        formatted = []
        for i, result in enumerate(search_results, 1):
            title = self.content_processor.sanitize_content(
                str(result.get("title", "No title"))
            )
            content = self.content_processor.sanitize_content(
                str(result.get("content", result.get("snippet", "No content")))
            )
            url = self.content_processor.sanitize_content(
                str(result.get("url", "No link"))
            )

            formatted_result = f"""Result {i}:
Title: {title}
Link: {url}
Content: {content}
---"""
            formatted.append(formatted_result)

        return "\n\n".join(formatted)

    def _process_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean search results with advanced cleaning"""
        processed_results = []
        
        for i, result in enumerate(search_results):
            # 获取原始内容
            raw_title = str(result.get("title", f"Result {i+1}"))
            raw_content = str(result.get("content", result.get("snippet", "")))
            raw_url = str(result.get("url", ""))
            
            # 使用新的清洗器进行深度清洗
            cleaned_title = self.cleaner.clean_text(raw_title)
            cleaned_content = self.cleaner.clean_text(raw_content)
            
            # 验证URL有效性
            if not self.cleaner.is_valid_url(raw_url):
                continue  # 跳过无效URL的结果
            
            # 计算内容质量分数
            quality_score = self.cleaner.calculate_content_quality(cleaned_content)
            
            # 过滤低质量内容
            if quality_score < 0.3 or not cleaned_content:
                continue
            
            # 提取关键词和关键点
            keywords = self.cleaner.extract_keywords(cleaned_content, 8)
            key_points = self._extract_enhanced_key_points(cleaned_content, keywords)
            
            # 最终清理（使用原有的sanitize_content作为补充）
            title = self.content_processor.sanitize_content(cleaned_title)
            content = self.content_processor.sanitize_content(cleaned_content)
            url = self.content_processor.sanitize_content(raw_url)
            
            # 创建处理后的结果
            processed_result = {
                "title": title,
                "content": content,
                "url": url,
                "relevance_score": quality_score,
                "key_points": key_points,
                "keywords": keywords[:5],  # 保留前5个关键词
                "content_length": len(content),
                "quality_score": quality_score
            }
            
            processed_results.append(processed_result)
        
        # 按质量分数排序
        processed_results.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return processed_results

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content using simple text processing"""
        if not content:
            return []
        
        # Simple key point extraction - split by sentences and take first few
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        # Return first 3 sentences as key points, limited to reasonable length
        key_points = []
        for sentence in sentences[:3]:
            if len(sentence) > 10 and len(sentence) < 200:  # Filter reasonable length sentences
                key_points.append(sentence)
        
        return key_points[:3]  # Maximum 3 key points
    
    def _extract_enhanced_key_points(self, content: str, keywords: List[str]) -> List[str]:
        """Enhanced key point extraction using keywords and sentence scoring"""
        if not content:
            return []
        
        # 分割句子
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return []
        
        # 为每个句子计算分数
        sentence_scores = []
        for sentence in sentences:
            if len(sentence) < 15 or len(sentence) > 300:
                continue
            
            score = 0
            sentence_lower = sentence.lower()
            
            # 关键词匹配分数
            keyword_matches = sum(1 for keyword in keywords if keyword in sentence_lower)
            score += keyword_matches * 2
            
            # 句子位置分数（开头的句子更重要）
            position_score = max(0, 3 - sentences.index(sentence) * 0.5)
            score += position_score
            
            # 句子长度分数（适中长度更好）
            length = len(sentence)
            if 50 <= length <= 150:
                score += 2
            elif 150 < length <= 250:
                score += 1
            
            # 信息密度分数（包含数字、专业术语等）
            if re.search(r'\d+', sentence):  # 包含数字
                score += 0.5
            if re.search(r'[A-Z]{2,}', sentence):  # 包含缩写
                score += 0.5
            
            sentence_scores.append((sentence, score))
        
        # 按分数排序并返回前3个
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        key_points = [sentence for sentence, score in sentence_scores[:3] if score > 0]
        
        return key_points

    def batch_clean_search_results(self, search_results_list: List[List[Dict[str, Any]]], 
                                  model_name: str) -> List[Dict[str, Any]]:
        """批量清洗多个搜索结果集
        
        Args:
            search_results_list: 多个搜索结果集的列表
            model_name: 模型名称
            
        Returns:
            List[Dict[str, Any]]: 批量处理后的结果列表
        """
        batch_results = []
        
        for i, search_results in enumerate(search_results_list):
            logger.debug(f"Processing batch {i+1}/{len(search_results_list)}")
            
            # 应用配置限制
            max_results = self.cleaning_config.get('max_results', 20)
            limited_results = search_results[:max_results]
            
            # 处理单个结果集
            processed_batch = self._process_search_results_with_config(limited_results)
            
            batch_results.append({
                'batch_id': i,
                'original_count': len(search_results),
                'processed_count': len(processed_batch),
                'results': processed_batch
            })
        
        return batch_results
    
    def _process_search_results_with_config(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用配置选项处理搜索结果"""
        processed_results = []
        
        for i, result in enumerate(search_results):
            # 获取原始内容，优先使用content而非raw_content
            raw_title = str(result.get("title", f"Result {i+1}"))
            # 优先使用content字段，避免raw_content中的无用信息
            raw_content = str(result.get("content", result.get("snippet", "")))
            raw_url = str(result.get("url", ""))
            
            # 如果存在raw_content但没有content，则从raw_content中提取有用信息
            if not raw_content and result.get("raw_content"):
                raw_content = self._extract_useful_content_from_raw(result.get("raw_content"))
            
            # 使用清洗器进行深度清洗
            cleaned_title = self.cleaner.clean_text(raw_title)
            cleaned_content = self.cleaner.clean_text(raw_content)
            
            # 根据配置验证URL有效性
            if self.cleaning_config.get('filter_invalid_urls', True):
                if not self.cleaner.is_valid_url(raw_url):
                    continue
            
            # 计算内容质量分数
            quality_score = self.cleaner.calculate_content_quality(cleaned_content)
            
            # 根据配置过滤低质量内容
            min_quality = self.cleaning_config.get('min_quality_score', 0.3)
            if quality_score < min_quality or not cleaned_content:
                continue
            
            # 根据配置提取关键词
            keywords = []
            if self.cleaning_config.get('enable_keyword_extraction', True):
                keywords = self.cleaner.extract_keywords(cleaned_content, 8)
            
            # 根据配置提取关键点
            key_points = []
            if self.cleaning_config.get('enable_enhanced_key_points', True):
                key_points = self._extract_enhanced_key_points(cleaned_content, keywords)
            else:
                key_points = self._extract_key_points(cleaned_content)
            
            # 最终清理
            title = self.content_processor.sanitize_content(cleaned_title)
            content = self.content_processor.sanitize_content(cleaned_content)
            url = self.content_processor.sanitize_content(raw_url)
            
            # 创建处理后的结果，不包含raw_content
            processed_result = {
                "title": title,
                "content": content,
                "url": url,
                "relevance_score": quality_score,
                "key_points": key_points,
                "keywords": keywords[:5],
                "content_length": len(content),
                "quality_score": quality_score
            }
            
            processed_results.append(processed_result)
        
        # 根据配置按质量分数排序
        if self.cleaning_config.get('sort_by_quality', True):
            processed_results.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return processed_results

    def format_filtered_results(self, filtered_data: Dict[str, Any]) -> str:
        """Format filtered results as text"""
        if not filtered_data.get("filtered_results"):
            return filtered_data.get("summary", "No relevant information found")

        formatted_parts = []

        # Add summary
        if filtered_data.get("summary"):
            formatted_parts.append(
                f"## Search Results Summary\n\n{filtered_data['summary']}\n"
            )

        # Add filtered results
        formatted_parts.append("## Relevant Information\n")

        for i, result in enumerate(filtered_data["filtered_results"], 1):
            title = result.get("title", f"Result {i}")
            content = result.get("content", "")
            url = result.get("url", "")
            relevance = result.get("relevance_score", "N/A")

            result_text = f"### {i}. {title}\n\n"
            if content:
                result_text += f"{content}\n\n"
            if url and url != "No link":
                result_text += f"**Source**: {url}\n\n"
            if relevance != "N/A":
                result_text += f"**Relevance**: {relevance}\n\n"

            formatted_parts.append(result_text)

        # Add statistics
        total_filtered = filtered_data.get("total_filtered", 0)
        total_original = filtered_data.get("total_original", 0)
        formatted_parts.append(
            f"---\n\n*Filtering Statistics: {total_filtered} relevant results filtered from {total_original} total results*"
        )

        return "\n".join(formatted_parts)
    
    def _extract_useful_content_from_raw(self, raw_content: str) -> str:
        """从raw_content中提取有用信息，过滤掉广告、外链等无用内容"""
        if not raw_content:
            return ""
        
        # 移除HTML标签
        import re
        text = re.sub(r'<[^>]+>', ' ', raw_content)
        
        # 移除常见的广告和无用信息模式
        ad_patterns = [
            r'广告[^\n]*',  # 广告相关
            r'Advertisement[^\n]*',
            r'Sponsored[^\n]*',
            r'推广[^\n]*',
            r'免责声明[^\n]*',
            r'Disclaimer[^\n]*',
            r'版权所有[^\n]*',
            r'Copyright[^\n]*',
            r'All rights reserved[^\n]*',
            r'更多.*?请访问[^\n]*',
            r'点击.*?了解更多[^\n]*',
            r'立即.*?咨询[^\n]*',
            r'联系我们[^\n]*',
            r'Contact us[^\n]*',
            r'订阅[^\n]*',
            r'Subscribe[^\n]*',
            r'关注我们[^\n]*',
            r'Follow us[^\n]*',
            r'分享到[^\n]*',
            r'Share[^\n]*',
            r'相关推荐[^\n]*',
            r'Related[^\n]*',
            r'热门.*?推荐[^\n]*',
            r'Popular[^\n]*',
            r'猜你喜欢[^\n]*',
            r'You might like[^\n]*'
        ]
        
        for pattern in ad_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除外链模式
        link_patterns = [
            r'http[s]?://[^\s]+',  # HTTP链接
            r'www\.[^\s]+',  # www链接
            r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # 域名
            r'点击.*?链接[^\n]*',
            r'Click.*?link[^\n]*',
            r'访问.*?网站[^\n]*',
            r'Visit.*?website[^\n]*'
        ]
        
        for pattern in link_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除图片相关信息
        image_patterns = [
            r'\[图片\]',
            r'\[Image\]',
            r'\[img\]',
            r'图片来源[^\n]*',
            r'Image source[^\n]*',
            r'图：[^\n]*',
            r'Figure[^\n]*'
        ]
        
        for pattern in image_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 移除开头和结尾的空白
        text = text.strip()
        
        # 如果处理后的文本太短，返回空字符串
        if len(text) < 20:
            return ""
        
        return text
