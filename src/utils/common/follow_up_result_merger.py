"""Follow-up查询结果合并机制

这个模块提供了智能的Follow-up查询结果合并功能，包括：
- 内容去重和相似性检测
- 智能结果优先级排序
- 结构化数据合并
- 质量评估和过滤
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """合并后的结果数据结构"""

    content: str
    sources: List[str]
    confidence_score: float
    relevance_score: float
    key_points: List[str]
    metadata: Dict[str, Any]
    original_count: int
    merged_count: int


@dataclass
class ResultMetrics:
    """结果质量指标"""

    content_length: int
    unique_information_ratio: float
    source_diversity: int
    temporal_relevance: float
    structural_completeness: float


class FollowUpResultMerger:
    """Follow-up查询结果智能合并器"""

    def __init__(
        self,
        config: Optional[Any] = None,
        similarity_threshold: Optional[float] = None,
        min_content_length: Optional[int] = None,
        max_merged_results: Optional[int] = None,
        enable_semantic_grouping: Optional[bool] = None,
    ):
        """
        初始化合并器

        Args:
            config: 合并器配置对象（优先级最高）
            similarity_threshold: 内容相似度阈值（向后兼容）
            min_content_length: 最小内容长度（向后兼容）
            max_merged_results: 最大合并结果数（向后兼容）
            enable_semantic_grouping: 是否启用语义分组（向后兼容）
        """
        # 导入配置（延迟导入避免循环依赖）
        if config is None:
            try:
                # 优先使用新的统一配置系统
                from src.config.config_loader import get_settings
                app_settings = get_settings()
                config = app_settings.get_followup_merger_config()
            except ImportError:
                try:
                    # 回退到旧的配置系统
                    from src.config.follow_up_merger_config import get_active_merger_config
                    config = get_active_merger_config()
                except ImportError:
                    # 如果配置模块不可用，使用默认值
                    from src.config.follow_up_merger_config import FollowUpMergerConfig
                    config = FollowUpMergerConfig()

        # 应用参数覆盖（向后兼容）
        self.config = config
        if similarity_threshold is not None:
            self.config.similarity_threshold = similarity_threshold
        if min_content_length is not None:
            self.config.min_content_length = min_content_length
        if max_merged_results is not None:
            self.config.max_merged_results = max_merged_results
        if enable_semantic_grouping is not None:
            self.config.enable_semantic_grouping = enable_semantic_grouping

        # 配置已在 Pydantic 模型中自动验证

        # 内容指纹缓存
        self._content_fingerprints: Set[str] = set()
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

        # 性能统计
        self._stats = {
            "total_merges": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_count": 0,
            "quality_filtered_count": 0,
        }

    def merge_follow_up_results(
        self,
        follow_up_results: List[Dict[str, Any]],
        original_findings: List[str],
        query_context: Optional[str] = None,
    ) -> List[MergedResult]:
        """
        合并Follow-up查询结果

        Args:
            follow_up_results: Follow-up查询的原始结果
            original_findings: 原始研究发现
            query_context: 查询上下文

        Returns:
            合并后的结果列表
        """
        logger.info(f"开始合并 {len(follow_up_results)} 个Follow-up结果")

        # 更新统计信息
        self._stats["total_merges"] += 1

        # 1. 预处理和标准化
        normalized_results = self._normalize_results(follow_up_results)

        # 2. 内容去重
        deduplicated_results = self._deduplicate_content(normalized_results)

        # 3. 与原始发现去重
        filtered_results = self._filter_against_original(
            deduplicated_results, original_findings
        )

        # 4. 语义分组（如果启用）
        if self.config.enable_semantic_grouping:
            grouped_results = self._group_by_semantic_similarity(filtered_results)
        else:
            grouped_results = [[result] for result in filtered_results]

        # 5. 组内合并
        merged_groups = []
        for group in grouped_results:
            merged_result = self._merge_group(group, query_context)
            if merged_result:
                merged_groups.append(merged_result)

        # 6. 质量评估和排序
        scored_results = self._score_and_rank_results(merged_groups)

        # 7. 最终过滤和限制数量
        final_results = self._apply_final_filters(scored_results)

        logger.info(
            f"合并完成，从 {len(follow_up_results)} 个结果合并为 {len(final_results)} 个"
        )
        return final_results

    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化结果格式"""
        normalized = []

        for i, result in enumerate(results):
            # 提取内容
            content = ""
            if isinstance(result, dict):
                content = result.get("content", "")
                if not content:
                    # 尝试其他字段
                    content = result.get("observation", "")
                    if not content and "update" in result:
                        update = result["update"]
                        if isinstance(update, dict):
                            observations = update.get("observations", [])
                            if observations:
                                content = " ".join(
                                    (
                                        obs.get("content", str(obs))
                                        if isinstance(obs, dict)
                                        else str(obs)
                                    )
                                    for obs in observations
                                )
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)

            # 清理内容
            content = self._clean_content(content)

            if len(content) >= self.config.min_content_length:
                normalized.append(
                    {
                        "content": content,
                        "source": (
                            result.get("source", f"follow_up_{i+1}")
                            if isinstance(result, dict)
                            else f"follow_up_{i+1}"
                        ),
                        "metadata": result if isinstance(result, dict) else {},
                        "original_index": i,
                    }
                )

        return normalized

    def _clean_content(self, content: str) -> str:
        """清理内容文本"""
        if not content:
            return ""

        # 移除Follow-up标记
        content = re.sub(r"\[Follow-up \d+\.\d+\]\s*", "", content)

        # 移除多余的空白字符
        content = re.sub(r"\s+", " ", content).strip()

        # 移除重复的句子开头
        lines = content.split("\n")
        cleaned_lines = []
        prev_line = ""

        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line

        return "\n".join(cleaned_lines)

    def _deduplicate_content(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """基于内容指纹去重"""
        deduplicated = []
        seen_fingerprints = set()

        for result in results:
            content = result["content"]
            fingerprint = self._generate_content_fingerprint(content)

            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                deduplicated.append(result)
            else:
                logger.debug(f"发现重复内容，已跳过: {content[:100]}...")

        logger.info(f"去重后保留 {len(deduplicated)}/{len(results)} 个结果")
        return deduplicated

    def _generate_content_fingerprint(self, content: str) -> str:
        """生成内容指纹"""
        # 标准化文本
        normalized = re.sub(r"\W+", " ", content.lower()).strip()

        # 提取关键词（去除停用词）
        words = normalized.split()
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "是",
            "的",
            "了",
            "在",
            "有",
            "和",
            "与",
        }
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # 取前20个关键词生成指纹
        key_content = " ".join(sorted(keywords[:20]))
        return hashlib.md5(key_content.encode()).hexdigest()

    def _filter_against_original(
        self, results: List[Dict[str, Any]], original_findings: List[str]
    ) -> List[Dict[str, Any]]:
        """过滤与原始发现重复的内容"""
        if not original_findings:
            return results

        # 检查是否与原始发现重复
        original_fingerprints = set()
        for finding in original_findings:
            if (
                isinstance(finding, str)
                and len(finding) >= self.config.min_content_length
            ):
                fingerprint = self._generate_content_fingerprint(finding)
                original_fingerprints.add(fingerprint)

        # 过滤重复内容
        filtered = []
        for result in results:
            content_fingerprint = self._generate_content_fingerprint(result["content"])

            # 检查是否与原始发现重复
            is_duplicate = content_fingerprint in original_fingerprints

            # 检查相似度
            if not is_duplicate:
                max_similarity = 0.0
                for finding in original_findings:
                    if isinstance(finding, str):
                        similarity = self._calculate_similarity(
                            result["content"], finding
                        )
                        max_similarity = max(max_similarity, similarity)

                is_duplicate = max_similarity > self.config.similarity_threshold

            if not is_duplicate:
                filtered.append(result)
            else:
                self._stats["deduplication_count"] += 1
                logger.debug(f"过滤重复内容: {result['content'][:100]}...")

        logger.info(f"原始发现过滤后保留 {len(filtered)}/{len(results)} 个结果")
        return filtered

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（带缓存）
        """
        # 创建缓存键
        cache_key = (hash(text1), hash(text2))
        if cache_key[0] > cache_key[1]:
            cache_key = (cache_key[1], cache_key[0])

        # 检查缓存
        if cache_key in self._similarity_cache:
            self._stats["cache_hits"] += 1
            return self._similarity_cache[cache_key]

        self._stats["cache_misses"] += 1

        try:
            # 使用简单的词汇重叠度计算
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0

            # 缓存结果
            self._similarity_cache[cache_key] = similarity

            # 限制缓存大小
            if len(self._similarity_cache) > 1000:
                # 清理最旧的一半缓存
                keys_to_remove = list(self._similarity_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._similarity_cache[key]

            return similarity

        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            return 0.0

    def _group_by_semantic_similarity(
        self, results: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """基于语义相似性分组"""
        if not results:
            return []

        groups = []
        ungrouped = results.copy()

        while ungrouped:
            # 取第一个作为组的种子
            seed = ungrouped.pop(0)
            current_group = [seed]

            # 找到相似的结果
            remaining = []
            for result in ungrouped:
                similarity = self._calculate_similarity(
                    seed["content"], result["content"]
                )

                if similarity > self.config.similarity_threshold:
                    current_group.append(result)
                else:
                    remaining.append(result)

            ungrouped = remaining
            groups.append(current_group)

        logger.info(f"语义分组完成: {len(results)} 个结果分为 {len(groups)} 组")
        return groups

    def _merge_group(
        self, group: List[Dict[str, Any]], query_context: Optional[str] = None
    ) -> Optional[MergedResult]:
        """合并同组内的结果"""
        if not group:
            return None

        if len(group) == 1:
            # 单个结果直接转换
            result = group[0]
            return MergedResult(
                content=result["content"],
                sources=[result["source"]],
                confidence_score=0.8,
                relevance_score=0.7,
                key_points=self._extract_key_points(result["content"]),
                metadata=result["metadata"],
                original_count=1,
                merged_count=1,
            )

        # 多个结果需要合并
        merged_content = self._merge_content([r["content"] for r in group])
        sources = [r["source"] for r in group]

        # 合并元数据
        merged_metadata = {}
        for result in group:
            merged_metadata.update(result["metadata"])

        # 计算置信度和相关性分数
        confidence_score = min(0.9, 0.6 + len(group) * 0.1)  # 更多来源 = 更高置信度
        relevance_score = self._calculate_relevance_score(merged_content, query_context)

        return MergedResult(
            content=merged_content,
            sources=sources,
            confidence_score=confidence_score,
            relevance_score=relevance_score,
            key_points=self._extract_key_points(merged_content),
            metadata=merged_metadata,
            original_count=len(group),
            merged_count=1,
        )

    def _merge_content(self, contents: List[str]) -> str:
        """智能合并多个内容"""
        if not contents:
            return ""

        if len(contents) == 1:
            return contents[0]

        # 按长度排序，优先保留更详细的内容
        sorted_contents = sorted(contents, key=len, reverse=True)

        # 提取所有句子
        all_sentences = []
        for content in sorted_contents:
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            all_sentences.extend(sentences)

        # 去重句子（保持顺序）
        unique_sentences = []
        seen_sentences = set()

        for sentence in all_sentences:
            sentence_key = sentence.lower().strip()
            if sentence_key not in seen_sentences and len(sentence) > 10:
                seen_sentences.add(sentence_key)
                unique_sentences.append(sentence)

        # 重新组织内容
        merged = ". ".join(unique_sentences)
        if merged and not merged.endswith("."):
            merged += "."

        return merged

    def _extract_key_points(self, content: str) -> List[str]:
        """提取关键点"""
        if not content:
            return []

        sentences = [s.strip() for s in content.split(".") if s.strip()]

        # 简单的关键点提取：选择较长且信息丰富的句子
        key_points = []
        for sentence in sentences[:5]:  # 最多5个关键点
            if (
                len(sentence) > 20
                and len(sentence) < 200
                and any(char.isdigit() or char.isupper() for char in sentence)
            ):
                key_points.append(sentence)

        return key_points[:3]  # 最多3个关键点

    def _calculate_relevance_score(
        self, content: str, query_context: Optional[str] = None
    ) -> float:
        """计算相关性分数"""
        if not query_context:
            return 0.7  # 默认分数

        # 简单的关键词匹配
        content_lower = content.lower()
        context_lower = query_context.lower()

        # 提取关键词
        context_words = set(re.findall(r"\w+", context_lower))
        content_words = set(re.findall(r"\w+", content_lower))

        if not context_words:
            return 0.7

        # 计算交集比例
        intersection = context_words.intersection(content_words)
        relevance = len(intersection) / len(context_words)

        return min(1.0, relevance + 0.3)  # 基础分数 + 匹配度

    def _score_and_rank_results(
        self, results: List[MergedResult]
    ) -> List[MergedResult]:
        """评分和排序结果"""
        if not results:
            return []

        # 计算综合分数
        for result in results:
            # 综合分数 = 置信度 * 0.4 + 相关性 * 0.4 + 内容质量 * 0.2
            content_quality = self._calculate_content_quality(result.content)

            composite_score = (
                result.confidence_score * 0.4
                + result.relevance_score * 0.4
                + content_quality * 0.2
            )

            # 更新元数据
            result.metadata["composite_score"] = composite_score
            result.metadata["content_quality"] = content_quality

        # 按综合分数排序
        sorted_results = sorted(
            results, key=lambda x: x.metadata.get("composite_score", 0), reverse=True
        )

        return sorted_results

    def _calculate_content_quality(self, content: str) -> float:
        """计算内容质量分数"""
        if not content:
            return 0.0

        score = 0.0

        # 长度分数（适中长度更好）
        length = len(content)
        if 100 <= length <= 500:
            score += 0.3
        elif 500 < length <= 1000:
            score += 0.2
        elif length > 50:
            score += 0.1

        # 信息密度分数
        if re.search(r"\d+", content):  # 包含数字
            score += 0.2

        if re.search(r"[A-Z]{2,}", content):  # 包含缩写
            score += 0.1

        # 结构化程度
        sentences = content.split(".")
        if len(sentences) >= 3:
            score += 0.2

        # 专业术语密度
        words = content.split()
        long_words = [w for w in words if len(w) > 6]
        if len(long_words) / len(words) > 0.2:
            score += 0.2

        return min(1.0, score)

    def _apply_final_filters(self, results: List[MergedResult]) -> List[MergedResult]:
        """应用最终过滤器"""
        if not results:
            return []

        # 过滤低质量结果
        filtered = [
            result
            for result in results
            if result.metadata.get("content_quality", 0)
            >= self.config.quality_threshold
        ]

        # 统计过滤的数量
        self._stats["quality_filtered_count"] += len(results) - len(filtered)

        # 限制数量
        final_results = filtered[: self.config.max_merged_results]

        logger.info(
            f"最终过滤: {len(results)} -> {len(filtered)} -> {len(final_results)}"
        )
        return final_results

    def get_merge_statistics(self, results: List[MergedResult]) -> Dict[str, Any]:
        """获取合并统计信息"""
        if not results:
            return {}

        total_original = sum(r.original_count for r in results)
        total_merged = len(results)

        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)

        source_diversity = len(
            set(source for result in results for source in result.sources)
        )

        return {
            "total_original_results": total_original,
            "total_merged_results": total_merged,
            "compression_ratio": (
                total_original / total_merged if total_merged > 0 else 0
            ),
            "average_confidence": avg_confidence,
            "average_relevance": avg_relevance,
            "source_diversity": source_diversity,
            "quality_distribution": {
                "high": len(
                    [r for r in results if r.metadata.get("content_quality", 0) > 0.7]
                ),
                "medium": len(
                    [
                        r
                        for r in results
                        if 0.4 <= r.metadata.get("content_quality", 0) <= 0.7
                    ]
                ),
                "low": len(
                    [r for r in results if r.metadata.get("content_quality", 0) < 0.4]
                ),
            },
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        """
        return {
            "total_merges": self._stats["total_merges"],
            "cache_performance": {
                "hits": self._stats["cache_hits"],
                "misses": self._stats["cache_misses"],
                "hit_rate": (
                    self._stats["cache_hits"]
                    / (self._stats["cache_hits"] + self._stats["cache_misses"])
                    if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                    else 0
                ),
                "cache_size": len(self._similarity_cache),
            },
            "filtering_stats": {
                "deduplication_count": self._stats["deduplication_count"],
                "quality_filtered_count": self._stats["quality_filtered_count"],
            },
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "min_content_length": self.config.min_content_length,
                "max_merged_results": self.config.max_merged_results,
                "enable_semantic_grouping": self.config.enable_semantic_grouping,
                "enable_intelligent_merging": self.config.enable_intelligent_merging,
                "enable_deduplication": self.config.enable_deduplication,
                "enable_quality_filtering": self.config.enable_quality_filtering,
            },
        }

    def reset_stats(self):
        """
        重置性能统计信息
        """
        self._stats = {
            "total_merges": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_count": 0,
            "quality_filtered_count": 0,
        }
        self._similarity_cache.clear()
        self._content_fingerprints.clear()
