"""Follow-up Query Result Merger Configuration

This module defines configuration options for merging follow-up query results,
allowing users to customize merging behavior and parameters.
"""

from typing import Optional
from pydantic import BaseModel, Field


class FollowUpMergerConfig(BaseModel):
    """Configuration for merging follow-up query results"""

    # Basic merge parameters
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for determining if two results are similar",
    )

    min_content_length: int = Field(
        default=50,
        ge=10,
        description="Minimum content length; content shorter than this will be filtered",
    )

    max_merged_results: int = Field(
        default=20, ge=1, le=100, description="Maximum number of merged results"
    )

    # Feature toggles
    enable_semantic_grouping: bool = Field(
        default=True, description="Enable semantic grouping"
    )

    enable_intelligent_merging: bool = Field(
        default=True,
        description="Enable intelligent merging (fallback to simple concatenation if disabled)",
    )

    enable_deduplication: bool = Field(default=True, description="Enable deduplication")

    enable_quality_filtering: bool = Field(
        default=True, description="Enable quality filtering"
    )

    quality_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Quality threshold; content below this score will be filtered",
    )

    confidence_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight of confidence in the overall score",
    )

    relevance_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="weight of relevance"
    )

    content_quality_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="weight of content quality"
    )

    # Merge strategy parameters
    max_sentences_per_result: int = Field(
        default=10, ge=1, le=50, description="max sentences per result"
    )

    max_key_points: int = Field(
        default=3, ge=1, le=10, description="max key points per result"
    )

    preserve_source_info: bool = Field(
        default=True, description="check if preserve the source info"
    )

    # Performance optimization parameters
    enable_similarity_cache: bool = Field(
        default=True, description="Enable semantic grouping"
    )

    max_cache_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum size of the similarity cache",
    )

    # Logging and debugging
    enable_detailed_logging: bool = Field(
        default=False, description="Enable detailed logging"
    )

    log_merge_statistics: bool = Field(
        default=True, description="Whether to log merge statistics"
    )

    def validate_weights(self) -> "FollowUpMergerConfig":
        """Validate the sum of weights"""
        total_weight = (
            self.confidence_weight + self.relevance_weight + self.content_quality_weight
        )

        if abs(total_weight - 1.0) > 0.01:
            self.confidence_weight /= total_weight
            self.relevance_weight /= total_weight
            self.content_quality_weight /= total_weight

        return self

    @classmethod
    def create_conservative_config(cls) -> "FollowUpMergerConfig":
        """Create a conservative merge configuration (fewer merges, higher quality requirements)"""
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
        """Create an aggressive merge configuration (more merges, lower quality requirements)"""
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
        """Create a performance-optimized configuration (fast processing, basic features)"""
        return cls(
            similarity_threshold=0.7,
            min_content_length=50,
            max_merged_results=15,
            quality_threshold=0.3,
            enable_semantic_grouping=False,
            enable_intelligent_merging=True,
            enable_deduplication=True,
            enable_quality_filtering=False,
            enable_similarity_cache=True,
            max_cache_size=2000,
        )

    @classmethod
    def create_quality_focused_config(cls) -> "FollowUpMergerConfig":
        """Create a quality-focused configuration (highest quality results)"""
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
            content_quality_weight=0.4,
            max_sentences_per_result=8,
            max_key_points=5,
        )


class FollowUpMergerSettings(BaseModel):
    """Global settings for the follow-up merger"""

    default_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig, description="Default merge configuration"
    )

    conservative_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_conservative_config,
        description="Default merge configuration",
    )

    aggressive_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_aggressive_config,
        description="Conservative merge configuration",
    )

    performance_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_performance_config,
        description="Performance-optimized configuration",
    )

    quality_focused_config: FollowUpMergerConfig = Field(
        default_factory=FollowUpMergerConfig.create_quality_focused_config,
        description="Quality-focused configuration",
    )

    active_config_name: str = Field(
        default="default", description="Name of the currently active configuration"
    )

    enable_config_switching: bool = Field(
        default=True, description="Whether runtime configuration switching is allowed"
    )

    def get_active_config(self) -> FollowUpMergerConfig:
        """Get the currently active configuration"""
        config_map = {
            "default": self.default_config,
            "conservative": self.conservative_config,
            "aggressive": self.aggressive_config,
            "performance": self.performance_config,
            "quality_focused": self.quality_focused_config,
        }

        return config_map.get(self.active_config_name, self.default_config)

    def switch_config(self, config_name: str) -> bool:
        """Switch configuration"""
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
        """Get the list of available configurations"""
        return [
            "default",
            "conservative",
            "aggressive",
            "performance",
            "quality_focused",
        ]


_global_merger_settings: Optional[FollowUpMergerSettings] = None


def get_merger_settings() -> FollowUpMergerSettings:
    """Get global merger settings"""
    global _global_merger_settings
    if _global_merger_settings is None:
        _global_merger_settings = FollowUpMergerSettings()
    return _global_merger_settings


def set_merger_settings(settings: FollowUpMergerSettings) -> None:
    """Set global merger settings"""
    global _global_merger_settings
    _global_merger_settings = settings


def get_active_merger_config() -> FollowUpMergerConfig:
    """Get the currently active merger configuration"""
    return get_merger_settings().get_active_config()


def switch_merger_config(config_name: str) -> bool:
    """Switch merger configuration"""
    return get_merger_settings().switch_config(config_name)
