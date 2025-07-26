"""Configuration management module - Unified management of various researcher configurations"""

import logging
import types
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

    # Phase 4 reflection configuration
    enable_enhanced_reflection: bool = True
    max_reflection_loops: int = 3
    reflection_model: str = "basic"
    knowledge_gap_threshold: float = 0.7
    sufficiency_threshold: float = 0.8
    enable_reflection_integration: bool = True

    # Phase 5 iterative research configuration
    max_follow_up_iterations: int = 3
    enable_iterative_research: bool = True
    max_queries_per_iteration: int = 3
    follow_up_delay_seconds: float = 1.0

    # Tool configuration
    max_search_results: int = 10
    enable_smart_filtering: bool = True

    # MCP configuration
    mcp_enabled: bool = False
    mcp_servers: list = None

    # Parallel execution configuration
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = 3

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
                    new_configurable = types.SimpleNamespace(
                        **(
                            self.configurable.__dict__
                            if hasattr(self.configurable, "__dict__")
                            else {}
                        )
                    )
                    new_configurable.locale = state_locale
                    self.configurable = new_configurable

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
            reflection_model = getattr(
                self.configurable.reflection, "reflection_model", None
            )
        except AttributeError:
            pass

        if not reflection_model:
            # Determine reflection model based on deep thinking settings
            try:
                enable_deep_thinking = getattr(
                    self.configurable.agents, "enable_deep_thinking", False
                )
                reflection_model = "reasoning" if enable_deep_thinking else "basic"
            except AttributeError:
                reflection_model = "basic"

        reflection_config = {
            "enable_enhanced_reflection": getattr(
                self.configurable, "enable_enhanced_reflection", True
            ),
            "max_reflection_loops": getattr(
                self.configurable, "max_reflection_loops", 3
            ),
            "reflection_model": reflection_model,
            "knowledge_gap_threshold": getattr(
                self.configurable, "knowledge_gap_threshold", 0.7
            ),
            "sufficiency_threshold": getattr(
                self.configurable, "sufficiency_threshold", 0.8
            ),
            "enable_reflection_integration": getattr(
                self.configurable, "enable_reflection_integration", True
            ),
        }

        logger.info(
            f"Reflection config: enabled={reflection_config['enable_enhanced_reflection']}, "
            f"model={reflection_config['reflection_model']}"
        )
        return reflection_config

    def setup_iterative_research_config(self) -> Dict[str, Any]:
        """Set up Phase 5 iterative research configuration

        Returns:
            Iterative research configuration dictionary
        """
        logger.debug("Setting up iterative research configuration")

        iterative_config = {
            "max_follow_up_iterations": getattr(
                self.configurable, "max_follow_up_iterations", 3
            ),
            "enable_iterative_research": getattr(
                self.configurable, "enable_iterative_research", True
            ),
            "max_queries_per_iteration": getattr(
                self.configurable, "max_queries_per_iteration", 3
            ),
            "follow_up_delay_seconds": getattr(
                self.configurable, "follow_up_delay_seconds", 1.0
            ),
        }

        logger.info(
            f"Iterative research config: enabled={iterative_config['enable_iterative_research']}, "
            f"max_iterations={iterative_config['max_follow_up_iterations']}"
        )
        return iterative_config

    def setup_tool_config(self) -> Dict[str, Any]:
        """Set up tool configuration

        Returns:
            Tool configuration dictionary
        """
        logger.debug("Setting up tool configuration")

        tool_config = {
            "max_search_results": getattr(
                self.configurable.agents, "max_search_results", 10
            ),
            "enable_smart_filtering": getattr(
                self.configurable.content, "enable_smart_filtering", True
            ),
        }

        return tool_config

    def setup_mcp_config(self) -> Dict[str, Any]:
        """Set up MCP configuration

        Returns:
            MCP configuration dictionary
        """
        logger.debug("Setting up MCP configuration")

        mcp_config = {
            "enabled": getattr(self.configurable.mcp, "enabled", False),
            "servers": getattr(self.configurable.mcp, "servers", []),
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
        iterative_config = self.setup_iterative_research_config()
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
            # Phase 4 reflection configuration
            enable_enhanced_reflection=reflection_config["enable_enhanced_reflection"],
            max_reflection_loops=reflection_config["max_reflection_loops"],
            reflection_model=reflection_config["reflection_model"],
            knowledge_gap_threshold=reflection_config["knowledge_gap_threshold"],
            sufficiency_threshold=reflection_config["sufficiency_threshold"],
            enable_reflection_integration=reflection_config[
                "enable_reflection_integration"
            ],
            # Phase 5 iterative research configuration
            max_follow_up_iterations=iterative_config["max_follow_up_iterations"],
            enable_iterative_research=iterative_config["enable_iterative_research"],
            max_queries_per_iteration=iterative_config["max_queries_per_iteration"],
            follow_up_delay_seconds=iterative_config["follow_up_delay_seconds"],
            # Tool configuration
            max_search_results=tool_config["max_search_results"],
            enable_smart_filtering=tool_config["enable_smart_filtering"],
            # MCP configuration
            mcp_enabled=mcp_config["enabled"],
            mcp_servers=mcp_config["servers"],
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
        """Get reflection configuration dictionary (for backward compatibility)

        Returns:
            Reflection configuration dictionary
        """
        unified = self.get_unified_config()
        return {
            "enable_enhanced_reflection": unified.enable_enhanced_reflection,
            "max_reflection_loops": unified.max_reflection_loops,
            "reflection_model": unified.reflection_model,
            "knowledge_gap_threshold": unified.knowledge_gap_threshold,
            "sufficiency_threshold": unified.sufficiency_threshold,
            "enable_reflection_integration": unified.enable_reflection_integration,
        }
