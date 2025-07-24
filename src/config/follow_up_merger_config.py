"""Follow-up查询结果合并配置

这个模块定义了Follow-up查询结果合并的配置选项，
允许用户自定义合并行为和参数。
"""

from typing import Optional
from pydantic import BaseModel, Field


class FollowUpMergerConfig(BaseModel):
    """Follow-up查询结果合并配置"""

    # 基础合并参数
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="内容相似度阈值，用于判断两个结果是否相似",
    )

    min_content_length: int = Field(
        default=50, ge=10, description="最小内容长度，低于此长度的内容将被过滤"
    )

    max_merged_results: int = Field(
        default=20, ge=1, le=100, description="最大合并结果数量"
    )

    # 功能开关
    enable_semantic_grouping: bool = Field(default=True, description="是否启用语义分组")

    enable_intelligent_merging: bool = Field(
        default=True, description="是否启用智能合并（如果禁用，将使用简单拼接）"
    )

    enable_deduplication: bool = Field(default=True, description="是否启用去重功能")

    enable_quality_filtering: bool = Field(default=True, description="是否启用质量过滤")

    # 质量评估参数
    quality_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="内容质量阈值，低于此分数的内容将被过滤",
    )

    confidence_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="置信度在综合评分中的权重"
    )

    relevance_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="相关性在综合评分中的权重"
    )

    content_quality_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="内容质量在综合评分中的权重"
    )

    # 合并策略参数
    max_sentences_per_result: int = Field(
        default=10, ge=1, le=50, description="每个合并结果的最大句子数"
    )

    max_key_points: int = Field(
        default=3, ge=1, le=10, description="每个结果提取的最大关键点数"
    )

    preserve_source_info: bool = Field(default=True, description="是否保留源信息")

    # 性能优化参数
    enable_similarity_cache: bool = Field(
        default=True, description="是否启用相似度计算缓存"
    )

    max_cache_size: int = Field(
        default=1000, ge=100, le=10000, description="相似度缓存的最大大小"
    )

    # 日志和调试
    enable_detailed_logging: bool = Field(
        default=False, description="是否启用详细日志记录"
    )

    log_merge_statistics: bool = Field(default=True, description="是否记录合并统计信息")

    def validate_weights(self) -> "FollowUpMergerConfig":
        """验证权重总和"""
        total_weight = (
            self.confidence_weight + self.relevance_weight + self.content_quality_weight
        )

        if abs(total_weight - 1.0) > 0.01:  # 允许小的浮点误差
            # 自动标准化权重
            self.confidence_weight /= total_weight
            self.relevance_weight /= total_weight
            self.content_quality_weight /= total_weight

        return self

    @classmethod
    def create_conservative_config(cls) -> "FollowUpMergerConfig":
        """创建保守的合并配置（更少的合并，更高的质量要求）"""
        return cls(
            similarity_threshold=0.8,
            min_content_length=100,
            max_merged_results=10,
            quality_threshold=0.5,
            enable_semantic_grouping=True,
            enable_intelligent_merging=True,
            enable_deduplication=True,
            enable_quality_filtering=True,
        )

    @classmethod
    def create_aggressive_config(cls) -> "FollowUpMergerConfig":
        """创建激进的合并配置（更多的合并，更低的质量要求）"""
        return cls(
            similarity_threshold=0.5,
            min_content_length=30,
            max_merged_results=30,
            quality_threshold=0.2,
            enable_semantic_grouping=True,
            enable_intelligent_merging=True,
            enable_deduplication=True,
            enable_quality_filtering=True,
        )

    @classmethod
    def create_performance_config(cls) -> "FollowUpMergerConfig":
        """创建性能优化配置（快速处理，基础功能）"""
        return cls(
            similarity_threshold=0.7,
            min_content_length=50,
            max_merged_results=15,
            quality_threshold=0.3,
            enable_semantic_grouping=False,  # 禁用语义分组以提高性能
            enable_intelligent_merging=True,
            enable_deduplication=True,
            enable_quality_filtering=False,  # 禁用质量过滤以提高性能
            enable_similarity_cache=True,
            max_cache_size=2000,
        )

    @classmethod
    def create_quality_focused_config(cls) -> "FollowUpMergerConfig":
        """创建质量优先配置（最高质量的结果）"""
        return cls(
            similarity_threshold=0.8,
            min_content_length=80,
            max_merged_results=12,
            quality_threshold=0.6,
            enable_semantic_grouping=True,
            enable_intelligent_merging=True,
            enable_deduplication=True,
            enable_quality_filtering=True,
            confidence_weight=0.3,
            relevance_weight=0.3,
            content_quality_weight=0.4,  # 更重视内容质量
            max_sentences_per_result=8,
            max_key_points=5,
        )


class FollowUpMergerSettings(BaseModel):
    """Follow-up合并器全局设置"""

    # 默认配置
    default_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig, description="默认的合并配置"
    )

    # 预设配置
    conservative_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_conservative_config,
        description="保守的合并配置",
    )

    aggressive_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_aggressive_config,
        description="激进的合并配置",
    )

    performance_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_performance_config,
        description="性能优化配置",
    )

    quality_focused_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_quality_focused_config,
        description="质量优先配置",
    )

    # 运行时设置
    active_config_name: str = Field(default="default", description="当前激活的配置名称")

    enable_config_switching: bool = Field(
        default=True, description="是否允许运行时切换配置"
    )

    def get_active_config(self) -> FollowUpMergerConfig:
        """获取当前激活的配置"""
        config_map = {
            "default": self.default_config,
            "conservative": self.conservative_config,
            "aggressive": self.aggressive_config,
            "performance": self.performance_config,
            "quality_focused": self.quality_focused_config,
        }

        return config_map.get(self.active_config_name, self.default_config)

    def switch_config(self, config_name: str) -> bool:
        """切换配置"""
        if not self.enable_config_switching:
            return False

        valid_configs = {
            "default",
            "conservative",
            "aggressive",
            "performance",
            "quality_focused",
        }

        if config_name in valid_configs:
            self.active_config_name = config_name
            return True

        return False

    def get_available_configs(self) -> list[str]:
        """获取可用的配置列表"""
        return [
            "default",
            "conservative",
            "aggressive",
            "performance",
            "quality_focused",
        ]


# 全局设置实例
_global_merger_settings: Optional[FollowUpMergerSettings] = None


def get_merger_settings() -> FollowUpMergerSettings:
    """获取全局合并器设置"""
    global _global_merger_settings
    if _global_merger_settings is None:
        _global_merger_settings = FollowUpMergerSettings()
    return _global_merger_settings


def set_merger_settings(settings: FollowUpMergerSettings) -> None:
    """设置全局合并器设置"""
    global _global_merger_settings
    _global_merger_settings = settings


def get_active_merger_config() -> FollowUpMergerConfig:
    """获取当前激活的合并器配置"""
    return get_merger_settings().get_active_config()


def switch_merger_config(config_name: str) -> bool:
    """切换合并器配置"""
    return get_merger_settings().switch_config(config_name)
