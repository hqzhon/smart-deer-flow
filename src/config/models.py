"""
Pydantic-based configuration models for unified configuration management.
"""

from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class ReportStyle(str, Enum):
    """Report style enumeration."""

    ACADEMIC = "academic"
    BUSINESS = "business"
    TECHNICAL = "technical"
    CASUAL = "casual"


class SummaryType(str, Enum):
    """Summary type enumeration."""

    COMPREHENSIVE = "comprehensive"
    KEY_POINTS = "key_points"
    ABSTRACT = "abstract"


class IsolationLevel(str, Enum):
    """Researcher isolation level enumeration."""

    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class LLMModelConfig(BaseModel):
    """Individual LLM model configuration."""

    model: str
    api_key: str
    base_url: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    timeout: int = Field(default=30, ge=1)
    verify_ssl: bool = True


class LLMSettings(BaseModel):
    """LLM configuration settings."""

    basic_model: Optional[LLMModelConfig] = None
    reasoning_model: Optional[LLMModelConfig] = None
    reflection_model: Optional[LLMModelConfig] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    timeout: int = Field(default=30, ge=1)


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    connection_string: Optional[str] = None
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout: int = Field(default=30, ge=1)


class AgentSettings(BaseModel):
    """Agent configuration settings."""

    max_plan_iterations: int = Field(default=1, ge=1)
    max_step_num: int = Field(default=2, ge=1)
    max_search_results: int = Field(default=2, ge=1)
    enable_deep_thinking: bool = False
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = Field(default=3, ge=1)
    max_context_steps_parallel: int = Field(default=1, ge=1)
    disable_context_parallel: bool = False


class ResearchSettings(BaseModel):
    """Research-specific configuration settings."""

    enable_researcher_isolation: bool = True
    researcher_isolation_level: IsolationLevel = IsolationLevel.MODERATE
    researcher_max_local_context: int = Field(default=5000, ge=100)
    researcher_isolation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    researcher_auto_isolation: bool = False
    researcher_isolation_metrics: bool = False
    max_context_steps_researcher: int = Field(default=2, ge=1)


class ReflectionSettings(BaseModel):
    """Reflection system configuration settings."""

    # Core reflection control
    enable_enhanced_reflection: bool = Field(
        default=True, description="Enable enhanced reflection system"
    )
    max_reflection_loops: int = Field(
        default=1, ge=1, le=10, description="Maximum reflection loops per task"
    )
    max_total_reflections: int = Field(
        default=3, ge=1, le=50, description="Maximum total reflections per session"
    )
    reflection_temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Temperature for reflection model"
    )

    # Threshold settings
    reflection_confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence threshold for reflection"
    )
    knowledge_gap_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Knowledge gap threshold"
    )
    sufficiency_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Content sufficiency threshold"
    )

    # Integration settings
    enable_reflection_integration: bool = Field(
        default=True, description="Enable reflection integration with research flow"
    )
    enable_progressive_reflection: bool = Field(
        default=True, description="Enable progressive reflection across iterations"
    )
    enable_reflection_metrics: bool = Field(
        default=True, description="Enable reflection performance metrics"
    )
    skip_initial_stage_reflection: bool = Field(
        default=False, description="Skip reflection in initial research stage"
    )

    # Follow-up query control
    disable_followup_reflection: bool = Field(
        default=False, description="Disable reflection for follow-up queries"
    )
    merge_followup_results: bool = Field(
        default=True, description="Merge follow-up query results intelligently"
    )

    # Session management
    enable_global_counter: bool = Field(
        default=True, description="Enable global reflection counter across sessions"
    )
    reset_counter_on_new_task: bool = Field(
        default=True, description="Reset reflection counter for new tasks"
    )


class FollowUpMergerSettings(BaseModel):
    """Follow-up查询结果合并配置设置"""

    # 基础合并参数
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="内容相似度阈值"
    )
    min_content_length: int = Field(
        default=50, ge=10, description="最小内容长度"
    )
    max_merged_results: int = Field(
        default=20, ge=1, le=100, description="最大合并结果数量"
    )

    # 功能开关
    enable_semantic_grouping: bool = Field(
        default=True, description="启用语义分组"
    )
    enable_intelligent_merging: bool = Field(
        default=True, description="启用智能合并"
    )
    enable_deduplication: bool = Field(
        default=True, description="启用去重功能"
    )
    enable_quality_filtering: bool = Field(
        default=True, description="启用质量过滤"
    )

    # 质量评估参数
    quality_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="内容质量阈值"
    )
    confidence_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="置信度权重"
    )
    relevance_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="相关性权重"
    )
    content_quality_weight: float = Field(
        default=0.2, ge=0.0, le=1.0, description="内容质量权重"
    )

    # 合并策略参数
    max_sentences_per_result: int = Field(
        default=10, ge=1, le=50, description="每个结果最大句子数"
    )
    max_key_points: int = Field(
        default=2, ge=1, le=10, description="最大关键点数"
    )
    preserve_source_info: bool = Field(
        default=True, description="保留源信息"
    )

    # 性能优化参数
    enable_similarity_cache: bool = Field(
        default=True, description="启用相似度缓存"
    )
    max_cache_size: int = Field(
        default=1000, ge=100, le=10000, description="最大缓存大小"
    )

    # 预设配置选择
    active_config_preset: str = Field(
        default="default", description="当前激活的配置预设"
    )
    enable_config_switching: bool = Field(
        default=True, description="允许运行时切换配置"
    )

    # 日志和调试
    enable_detailed_logging: bool = Field(
        default=False, description="启用详细日志"
    )
    log_merge_statistics: bool = Field(
        default=True, description="记录合并统计信息"
    )

    @field_validator('confidence_weight', 'relevance_weight', 'content_quality_weight')
    @classmethod
    def validate_weights(cls, v, info):
        """验证权重值"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"权重值必须在0.0到1.0之间，当前值: {v}")
        return v

    def model_post_init(self, __context) -> None:
        """模型初始化后验证"""
        # 验证权重总和
        total_weight = self.confidence_weight + self.relevance_weight + self.content_quality_weight
        if abs(total_weight - 1.0) > 0.01:  # 允许小的浮点误差
            import warnings
            warnings.warn(
                f"权重总和应该接近1.0，当前总和: {total_weight:.3f}",
                UserWarning
            )





class IterativeResearchSettings(BaseModel):
    """Iterative research configuration settings."""

    max_follow_up_iterations: int = Field(default=1, ge=1)
    sufficiency_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_iterative_research: bool = True
    max_queries_per_iteration: int = Field(default=2, ge=1)
    follow_up_delay_seconds: float = Field(default=1.0, ge=0.0)


class ContentSettings(BaseModel):
    """Content processing configuration settings."""

    enable_content_summarization: bool = True
    enable_smart_filtering: bool = True
    summary_type: SummaryType = SummaryType.COMPREHENSIVE


class AdvancedContextConfig(BaseModel):
    """Advanced context management configuration."""

    max_context_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    sliding_window_size: int = Field(default=5, ge=1)
    overlap_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    compression_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    default_strategy: str = "adaptive"
    priority_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }
    )
    enable_caching: bool = True
    enable_analytics: bool = True
    debug_mode: bool = False


class MCPSettings(BaseModel):
    """MCP (Model Context Protocol) settings."""

    enabled: bool = False
    servers: List[Dict[str, Any]] = Field(default_factory=list)
    timeout: int = Field(default=30, ge=1)


class SearchEngine(str, Enum):
    """Search engine enumeration."""

    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    BRAVE_SEARCH = "brave_search"
    ARXIV = "arxiv"


class RAGProvider(str, Enum):
    """RAG provider enumeration."""

    RAGFLOW = "ragflow"


class ToolSettings(BaseModel):
    """Tool configuration settings."""

    search_engine: SearchEngine = SearchEngine.TAVILY
    rag_provider: Optional[RAGProvider] = None


class ConnectionPoolConfig(BaseModel):
    """Connection pool configuration."""

    max_connections: int = 50
    initial_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 500.0
    max_retries: int = 3


class BatchProcessingConfig(BaseModel):
    """Batch processing configuration."""

    batch_size: int = 10
    batch_timeout: float = 1.5
    max_queue_size: int = 1000
    priority_enabled: bool = True
    adaptive_sizing: bool = True


class CacheConfig(BaseModel):
    """Hierarchical cache configuration."""

    l1_size: int = 1000
    l2_size: int = 5000
    l3_size: int = 10000
    default_ttl: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    eviction_policy: str = "lru"  # lru, lfu, fifo


class RateLimitConfig(BaseModel):
    """Adaptive rate limiting configuration."""

    initial_rate: float = 10.0  # requests per second
    max_rate: float = 100.0
    min_rate: float = 1.0
    adaptation_factor: float = 1.2
    window_size: int = 60  # seconds
    time_window: int = 60  # seconds (alias for window_size for compatibility)
    burst_allowance: int = 20


class ErrorRecoveryConfig(BaseModel):
    """Smart error recovery configuration."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    jitter_enabled: bool = True


class ParallelExecutionConfig(BaseModel):
    """Advanced parallel execution configuration."""

    max_workers: int = 20
    queue_size: int = 1000
    priority_levels: int = 3
    load_balancing: bool = True
    worker_timeout: float = 500.0
    health_check_interval: float = 30.0


class MonitoringConfig(BaseModel):
    """Performance monitoring configuration."""

    metrics_enabled: bool = True
    detailed_logging: bool = True
    slow_request_threshold: float = 10.0  # seconds
    high_utilization_threshold: float = 0.8
    metrics_retention: int = 86400  # 24 hours
    export_interval: int = 60  # seconds


class PerformanceSettings(BaseModel):
    """Performance optimization configuration."""

    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)
    batch_processing: BatchProcessingConfig = Field(
        default_factory=BatchProcessingConfig
    )
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    error_recovery: ErrorRecoveryConfig = Field(default_factory=ErrorRecoveryConfig)
    parallel_execution: ParallelExecutionConfig = Field(
        default_factory=ParallelExecutionConfig
    )
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Global settings
    enable_advanced_optimization: bool = True
    enable_collaboration: bool = True
    debug_mode: bool = False


# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]


class AgentLLMSettings(BaseModel):
    """Agent-LLM mapping configuration."""

    coordinator: LLMType = "basic"
    planner: LLMType = "basic"
    researcher: LLMType = "basic"
    coder: LLMType = "basic"
    reporter: LLMType = "basic"
    podcast_script_writer: LLMType = "basic"
    ppt_composer: LLMType = "basic"
    prose_writer: LLMType = "basic"
    prompt_enhancer: LLMType = "basic"


class AppSettings(BaseSettings):
    """Main application configuration."""

    # Core settings
    report_style: ReportStyle = ReportStyle.ACADEMIC
    resources: List[Dict[str, Any]] = Field(default_factory=list)

    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    reflection: ReflectionSettings = Field(default_factory=ReflectionSettings)
    followup_merger: FollowUpMergerSettings = Field(default_factory=FollowUpMergerSettings)

    iterative_research: IterativeResearchSettings = Field(
        default_factory=IterativeResearchSettings
    )
    content: ContentSettings = Field(default_factory=ContentSettings)
    advanced_context: AdvancedContextConfig = Field(
        default_factory=AdvancedContextConfig
    )
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    agent_llm_map: AgentLLMSettings = Field(default_factory=AgentLLMSettings)

    # Model token limits
    model_token_limits: Dict[str, Dict[str, Union[int, float]]] = Field(
        default_factory=dict
    )

    model_config = SettingsConfigDict(
        env_prefix="DEER_",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow",
        env_nested_delimiter="__",
    )

    @field_validator("model_token_limits", mode="before")
    @classmethod
    def validate_token_limits(cls, v):
        """Validate model token limits format."""
        if isinstance(v, dict):
            return v
        return {}

    def get_llm_config(self) -> LLMSettings:
        """Get LLM-specific configuration."""
        return self.llm

    def get_agent_config(self) -> AgentSettings:
        """Get agent-specific configuration."""
        return self.agents

    def get_research_config(self) -> ResearchSettings:
        """Get research-specific configuration."""
        return self.research

    def get_reflection_config(self) -> ReflectionSettings:
        """Get reflection-specific configuration."""
        return self.reflection

    def get_followup_merger_config(self) -> FollowUpMergerSettings:
        """Get follow-up merger configuration."""
        return self.followup_merger



    def get_content_config(self) -> ContentSettings:
        """Get content-specific configuration."""
        return self.content

    def get_advanced_context_config(self) -> AdvancedContextConfig:
        """Get advanced context configuration."""
        return self.advanced_context

    def get_mcp_config(self) -> MCPSettings:
        """Get MCP configuration."""
        return self.mcp

    def get_tool_config(self) -> ToolSettings:
        """Get tool configuration."""
        return self.tools

    def get_performance_config(self) -> PerformanceSettings:
        """Get performance configuration."""
        return self.performance

    def get_agent_llm_config(self) -> AgentLLMSettings:
        """Get agent-LLM mapping configuration."""
        return self.agent_llm_map
