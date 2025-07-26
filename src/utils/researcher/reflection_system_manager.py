"""Reflection system management module - Unified management of reflection agents and integrators"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReflectionSystemManager:
    """Reflection system manager - Unified management of reflection agents and reflection integrators"""

    def __init__(self, unified_config: Any):
        """Initialize reflection system manager

        Args:
            unified_config: Unified configuration object
        """
        self.unified_config = unified_config
        self.reflection_agent = None
        self.reflection_integrator = None

        logger.info("ReflectionSystemManager initialized")

    async def setup_reflection_agent(self) -> Optional[Any]:
        """Set up reflection agent

        Returns:
            Reflection agent instance, or None if setup fails
        """
        if not self.unified_config.enable_enhanced_reflection:
            logger.info("Enhanced reflection is disabled")
            return None

        try:
            from src.utils.reflection.enhanced_reflection import EnhancedReflectionAgent

            # Extract configurable object from unified_config
            # This needs to be adjusted based on actual configuration structure
            configurable = getattr(self.unified_config, "configurable", None)
            if configurable is None:
                # If unified_config itself is a configurable object
                configurable = self.unified_config

            self.reflection_agent = EnhancedReflectionAgent(config=configurable)

            logger.info(
                f"Reflection agent initialized: model={self.unified_config.reflection_model}"
            )
            return self.reflection_agent

        except Exception as e:
            logger.error(f"Failed to initialize reflection agent: {e}")
            return None

    def setup_reflection_integrator(self, reflection_agent: Any) -> Optional[Any]:
        """Set up reflection integrator

        Args:
            reflection_agent: Reflection agent instance

        Returns:
            Reflection integrator instance, or None if setup fails
        """
        if not reflection_agent:
            logger.info("No reflection agent available, skipping integrator setup")
            return None

        try:
            from src.utils.reflection.reflection_integration import ReflectionIntegrator

            self.reflection_integrator = ReflectionIntegrator(
                reflection_agent=reflection_agent,
                config=self.unified_config,
            )

            logger.info("Reflection integrator initialized")
            return self.reflection_integrator

        except Exception as e:
            logger.error(f"Failed to initialize reflection integrator: {e}")
            return None

    async def initialize_reflection_system(self) -> tuple[Optional[Any], Optional[Any]]:
        """Initialize complete reflection system

        Returns:
            (reflection_agent, reflection_integrator) tuple
        """
        logger.info("Initializing reflection system")

        # Set up reflection agent
        reflection_agent = await self.setup_reflection_agent()

        # Set up reflection integrator
        reflection_integrator = self.setup_reflection_integrator(reflection_agent)

        if reflection_agent and reflection_integrator:
            logger.info("Reflection system initialized successfully")
        elif reflection_agent:
            logger.info("Reflection system partially initialized (agent only)")
        else:
            logger.info("Reflection system initialization skipped")

        return reflection_agent, reflection_integrator

    async def execute_reflection_analysis(self, reflection_context: Any) -> Any:
        """Execute reflection analysis

        Args:
            reflection_context: Reflection context

        Returns:
            Reflection result
        """
        if not self.reflection_agent:
            logger.warning("No reflection agent available for analysis")
            return None

        try:
            logger.info("Executing reflection analysis")

            reflection_result = await self.reflection_agent.analyze_knowledge_gaps(
                reflection_context,
                runnable_config=self.unified_config,
            )

            logger.info(
                f"Reflection analysis completed: sufficient={reflection_result.is_sufficient}, "
                f"confidence={reflection_result.confidence_score}, "
                f"gaps={len(reflection_result.knowledge_gaps) if reflection_result.knowledge_gaps else 0}"
            )

            return reflection_result

        except Exception as e:
            logger.error(f"Failed to execute reflection analysis: {e}")
            return None

    def should_trigger_reflection(
        self, state: Dict[str, Any], current_step: Any, agent_name: str
    ) -> tuple[bool, str, Dict[str, Any]]:
        """Check if reflection should be triggered

        Args:
            state: Current state
            current_step: Current step
            agent_name: Agent name

        Returns:
            (should_trigger, trigger_reason, decision_factors) tuple
        """
        if not self.reflection_integrator:
            return False, "No reflection integrator available", {}

        try:
            return self.reflection_integrator.should_trigger_reflection(
                state=state, current_step=current_step, agent_name=agent_name
            )
        except Exception as e:
            logger.error(f"Failed to check reflection trigger: {e}")
            return False, f"Error checking trigger: {e}", {}

    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get reflection system summary

        Returns:
            Reflection system summary dictionary
        """
        return {
            "reflection_enabled": self.unified_config.enable_enhanced_reflection,
            "reflection_model": self.unified_config.reflection_model,
            "max_reflection_loops": self.unified_config.max_reflection_loops,
            "knowledge_gap_threshold": self.unified_config.knowledge_gap_threshold,
            "sufficiency_threshold": self.unified_config.sufficiency_threshold,
            "reflection_integration_enabled": (
                self.unified_config.enable_reflection_integration
            ),
            "agent_initialized": self.reflection_agent is not None,
            "integrator_initialized": self.reflection_integrator is not None,
        }
