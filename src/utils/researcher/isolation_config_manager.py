"""Configuration management module - Unified management of various researcher configurations"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from langchain_core.runnables import RunnableConfig
from src.graph.nodes import get_configuration_from_config
from src.utils.researcher.researcher_context_isolator import ResearcherContextConfig

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResearchConfig:
    """Unified research configuration class"""

    # Language configuration
    locale: Optional[str] = None

    # Isolation configuration
    isolation_config: Optional[ResearcherContextConfig] = None

    # Phase 3 configuration
    researcher_isolation_metrics: bool = False
    researcher_auto_isolation: bool = False
    researcher_isolation_threshold: float = 0.7
    researcher_max_local_context: int = 3000

    # Unified reflection configuration (simplified)
    enabled: bool = True
    max_loops: int = 1
    quality_threshold: float = 0.7
    model: str = "basic"

    # Tool configuration
    max_search_results: int = 10
    enable_smart_filtering: bool = True

    # MCP configuration
    mcp_enabled: bool = False
    mcp_servers: list = None

    # Parallel execution configuration
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = 3
    max_context_steps_parallel: int = 1
    disable_context_parallel: bool = False

    # Other configuration
    max_step_num: Optional[int] = None

    def __post_init__(self):
        if self.mcp_servers is None:
            self.mcp_servers = []


class IsolationConfigManager:
    """Isolation configuration manager - Unified management of all configuration logic"""

    def __init__(self, state_or_configurable, config: RunnableConfig = None):
        """Initialize configuration manager

        Args:
            state_or_configurable: State object or configuration object
            config: Runtime configuration (optional)
        """
        if config is not None:
            # Full initialization mode: pass in state and config
            self.state = state_or_configurable
            self.config = config
            self.configurable = get_configuration_from_config(config)
        else:
            # Simplified initialization mode: only pass in configurable
            self.state = None
            self.config = None
            self.configurable = state_or_configurable

        self._unified_config = None

        logger.info("IsolationConfigManager initialized")

    def setup_language_config(self) -> Dict[str, Any]:
        """Set up language environment configuration

        Returns:
            Language configuration dictionary
        """
        logger.debug("Setting up language configuration")

        # Fix locale passing issue
        state_locale = None
        if self.state is not None:
            state_locale = self.state.get("locale")
            if state_locale and hasattr(self.configurable, "__dict__"):
                # Add locale to configurable if it exists in state
                self.configurable.locale = state_locale
            elif state_locale:
                # If configurable is a dict-like object, add locale
                if hasattr(self.configurable, "update"):
                    self.configurable.update({"locale": state_locale})
                else:
                    # Create a new configurable dict with locale
                    # Preserve the original configurable object and just set locale
                    if hasattr(self.configurable, "locale"):
                        self.configurable.locale = state_locale
                    else:
                        # Only create new namespace if configurable doesn't support locale
                        setattr(self.configurable, "locale", state_locale)

        return {"locale": state_locale, "configurable_updated": True}

    def setup_isolation_level(self) -> ResearcherContextConfig:
        """Set up isolation level configuration

        Returns:
            Isolation configuration object
        """
        logger.debug("Setting up isolation level configuration")

        isolation_level = getattr(
            self.configurable, "researcher_isolation_level", "moderate"
        )
        isolation_config = ResearcherContextConfig(
            max_context_steps=getattr(
                self.configurable, "max_context_steps_researcher", 2
            ),
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level=isolation_level,
        )

        logger.info(f"Isolation level configured: {isolation_level}")
        return isolation_config

    def setup_phase3_config(self) -> Dict[str, Any]:
        """Set up Phase 3 configuration (metrics and auto isolation)

        Returns:
            Phase 3 configuration dictionary
        """
        logger.debug("Setting up Phase 3 configuration")

        phase3_config = {
            "researcher_isolation_metrics": getattr(
                self.configurable, "researcher_isolation_metrics", False
            ),
            "researcher_auto_isolation": getattr(
                self.configurable, "researcher_auto_isolation", False
            ),
            "researcher_isolation_threshold": getattr(
                self.configurable, "researcher_isolation_threshold", 0.7
            ),
            "researcher_max_local_context": getattr(
                self.configurable, "researcher_max_local_context", 3000
            ),
        }

        logger.info(
            f"Phase 3 config: metrics={phase3_config['researcher_isolation_metrics']}, "
            f"auto_isolation={phase3_config['researcher_auto_isolation']}"
        )
        return phase3_config

    def setup_reflection_config(self) -> Dict[str, Any]:
        """Set up Phase 4 reflection configuration

        Returns:
            Reflection configuration dictionary
        """
        logger.debug("Setting up reflection configuration")

        # Get reflection model configuration
        reflection_model = None
        try:
            if (
                hasattr(self.configurable, "reflection")
                and self.configurable.reflection
            ):
                reflection_model = getattr(self.configurable.reflection, "model", None)
        except AttributeError:
            logger.warning("Could not access configurable.reflection")

        if not reflection_model:
            # Determine reflection model based on deep thinking settings
            try:
                enable_deep_thinking = False
                if hasattr(self.configurable, "agents") and self.configurable.agents:
                    enable_deep_thinking = getattr(
                        self.configurable.agents, "enable_deep_thinking", False
                    )
                reflection_model = "reasoning" if enable_deep_thinking else "basic"
            except AttributeError:
                logger.warning(
                    "Could not access configurable.agents, using basic reflection model"
                )
                reflection_model = "basic"

        # Get unified reflection configuration (simplified)
        reflection_attr = getattr(self.configurable, "reflection", None)
        if reflection_attr is None:
            # Create a default reflection configuration if not available
            reflection_config = {
                "enabled": True,
                "max_loops": 1,
                "quality_threshold": 0.7,
                "reflection_model": reflection_model,
            }
        else:
            reflection_config = {
                "enabled": getattr(reflection_attr, "enabled", True),
                "max_loops": getattr(reflection_attr, "max_loops", 1),
                "quality_threshold": getattr(reflection_attr, "quality_threshold", 0.7),
                "reflection_model": reflection_model,
            }

        logger.info(
            f"Unified reflection config: enabled={reflection_config['enabled']}, "
            f"model={reflection_config['reflection_model']}, "
            f"max_loops={reflection_config['max_loops']}"
        )
        return reflection_config

    def setup_tool_config(self) -> Dict[str, Any]:
        """Set up tool configuration

        Returns:
            Tool configuration dictionary
        """
        logger.debug("Setting up tool configuration")

        # Safely get max_search_results from agents
        max_search_results = 10
        try:
            if hasattr(self.configurable, "agents") and self.configurable.agents:
                max_search_results = getattr(
                    self.configurable.agents, "max_search_results", 10
                )
        except AttributeError:
            logger.warning(
                "Could not access configurable.agents, using default max_search_results=10"
            )

        # Safely get enable_smart_filtering from content
        enable_smart_filtering = True
        try:
            if hasattr(self.configurable, "content") and self.configurable.content:
                enable_smart_filtering = getattr(
                    self.configurable.content, "enable_smart_filtering", True
                )
        except AttributeError:
            logger.warning(
                "Could not access configurable.content, using default enable_smart_filtering=True"
            )

        tool_config = {
            "max_search_results": max_search_results,
            "enable_smart_filtering": enable_smart_filtering,
        }

        return tool_config

    def setup_mcp_config(self) -> Dict[str, Any]:
        """Set up MCP configuration

        Returns:
            MCP configuration dictionary
        """
        logger.debug("Setting up MCP configuration")

        # Safely get MCP configuration
        enabled = False
        servers = []
        try:
            if hasattr(self.configurable, "mcp") and self.configurable.mcp:
                enabled = getattr(self.configurable.mcp, "enabled", False)
                servers = getattr(self.configurable.mcp, "servers", [])
        except AttributeError:
            logger.warning(
                "Could not access configurable.mcp, using default MCP settings"
            )

        mcp_config = {
            "enabled": enabled,
            "servers": servers,
        }

        if mcp_config["enabled"]:
            logger.info(f"MCP enabled with {len(mcp_config['servers'])} servers")

        return mcp_config

    def get_unified_config(self) -> UnifiedResearchConfig:
        """Get unified configuration object

        Returns:
            Unified research configuration object
        """
        if self._unified_config is not None:
            return self._unified_config

        logger.info("Building unified configuration")

        # Set up various configuration modules
        language_config = self.setup_language_config()
        isolation_config = self.setup_isolation_level()
        phase3_config = self.setup_phase3_config()
        reflection_config = self.setup_reflection_config()
        tool_config = self.setup_tool_config()
        mcp_config = self.setup_mcp_config()

        # Build unified configuration
        self._unified_config = UnifiedResearchConfig(
            # Language configuration
            locale=language_config["locale"],
            # Isolation configuration
            isolation_config=isolation_config,
            # Phase 3 configuration
            researcher_isolation_metrics=phase3_config["researcher_isolation_metrics"],
            researcher_auto_isolation=phase3_config["researcher_auto_isolation"],
            researcher_isolation_threshold=phase3_config[
                "researcher_isolation_threshold"
            ],
            researcher_max_local_context=phase3_config["researcher_max_local_context"],
            # Unified reflection configuration (simplified)
            enabled=reflection_config["enabled"],
            max_loops=reflection_config["max_loops"],
            quality_threshold=reflection_config["quality_threshold"],
            model=reflection_config["reflection_model"],
            # Tool configuration
            max_search_results=tool_config["max_search_results"],
            enable_smart_filtering=tool_config["enable_smart_filtering"],
            # MCP configuration
            mcp_enabled=mcp_config["enabled"],
            mcp_servers=mcp_config["servers"],
            # Parallel execution configuration
            enable_parallel_execution=getattr(
                self.configurable, "enable_parallel_execution", True
            ),
            max_parallel_tasks=getattr(self.configurable, "max_parallel_tasks", 3),
            max_context_steps_parallel=getattr(
                self.configurable, "max_context_steps_parallel", 1
            ),
            disable_context_parallel=getattr(
                self.configurable, "disable_context_parallel", False
            ),
            # Other configuration
            max_step_num=getattr(self.configurable, "max_step_num", None),
        )

        logger.info("Unified configuration built successfully")
        return self._unified_config

    def get_phase3_config_dict(self) -> Dict[str, Any]:
        """Get Phase 3 configuration dictionary (for backward compatibility)

        Returns:
            Phase 3 configuration dictionary
        """
        unified = self.get_unified_config()
        return {
            "researcher_isolation_metrics": unified.researcher_isolation_metrics,
            "researcher_auto_isolation": unified.researcher_auto_isolation,
            "researcher_isolation_threshold": unified.researcher_isolation_threshold,
            "researcher_max_local_context": unified.researcher_max_local_context,
        }

    def get_reflection_config_dict(self) -> Dict[str, Any]:
        """Get unified reflection configuration dictionary (for backward compatibility)

        Returns:
            Unified reflection configuration dictionary
        """
        unified = self.get_unified_config()
        return {
            "enabled": unified.enabled,
            "max_loops": unified.max_loops,
            "quality_threshold": unified.quality_threshold,
            "reflection_model": unified.reflection_model,
        }

    async def setup_isolation_context(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set up isolation context for research execution

        Args:
            config: Optional configuration dictionary to override defaults

        Returns:
            Isolation context dictionary
        """
        logger.debug("Setting up isolation context")

        # Get unified configuration
        unified_config = self.get_unified_config()

        # Apply any config overrides
        if config:
            logger.debug(f"Applying config overrides: {list(config.keys())}")
            # Create a copy of unified config and update with overrides
            context_config = unified_config.__dict__.copy()
            context_config.update(config)
        else:
            context_config = unified_config.__dict__.copy()

        # Build isolation context
        isolation_context = {
            "isolation_config": unified_config.isolation_config,
            "language_config": {"locale": unified_config.locale},
            "tool_config": {
                "max_search_results": unified_config.max_search_results,
                "enable_smart_filtering": unified_config.enable_smart_filtering,
            },
            "reflection_config": {
                "enabled": unified_config.enabled,
                "max_loops": unified_config.max_loops,
                "quality_threshold": unified_config.quality_threshold,
                "reflection_model": unified_config.model,
            },
            "mcp_config": {
                "enabled": unified_config.mcp_enabled,
                "servers": unified_config.mcp_servers,
            },
            "phase3_config": {
                "researcher_isolation_metrics": (
                    unified_config.researcher_isolation_metrics
                ),
                "researcher_auto_isolation": unified_config.researcher_auto_isolation,
                "researcher_isolation_threshold": (
                    unified_config.researcher_isolation_threshold
                ),
                "researcher_max_local_context": (
                    unified_config.researcher_max_local_context
                ),
            },
            "parallel_config": {
                "enable_parallel_execution": unified_config.enable_parallel_execution,
                "max_parallel_tasks": unified_config.max_parallel_tasks,
                "max_context_steps_parallel": unified_config.max_context_steps_parallel,
                "disable_context_parallel": unified_config.disable_context_parallel,
            },
        }

        logger.info("Isolation context setup completed")
        return isolation_context

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging and monitoring

        Returns:
            Configuration summary dictionary
        """
        unified_config = self.get_unified_config()
        return {
            "locale": unified_config.locale,
            "isolation_level": (
                unified_config.isolation_config.isolation_level
                if unified_config.isolation_config
                else "default"
            ),
            "reflection_enabled": unified_config.enabled,
            "reflection_model": unified_config.model,
            "reflection_max_loops": unified_config.max_loops,
            "mcp_enabled": unified_config.mcp_enabled,
            "parallel_execution_enabled": unified_config.enable_parallel_execution,
            "max_search_results": unified_config.max_search_results,
        }
