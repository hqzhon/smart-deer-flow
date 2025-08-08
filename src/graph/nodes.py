# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Annotated, Literal, List, Dict, Any, Set

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

# Delayed import to avoid circular import
from src.agents import create_agent, create_agent_with_managed_prompt
from src.tools.search import LoggedTavilySearch
from src.tools import (
    python_repl_tool,
)
from src.tools.search import get_web_search_tool

from src.config.config_loader import get_settings
from src.config.models import SearchEngine
from src.config import get_settings as get_settings_alt

# ExecutionContextManager will be imported dynamically to avoid circular imports
from src.llms.error_handler import safe_llm_call, safe_llm_call_async
from src.llms.llm import get_llm_by_type
from src.utils.template import apply_prompt_template
from src.utils.common.json_utils import repair_json_output

from .types import State
from src.models.planner_model import Plan, Step, StepType

# ReflectionIntegrationConfig removed - using unified ReflectionSettings from src.config.models

logger = logging.getLogger(__name__)


def get_configuration_from_config(config):
    """Helper function to get configuration from settings with config overrides."""

    try:
        # Get base settings
        base_settings = get_settings_alt()

        # Create a configuration object that merges base settings with config overrides
        class ConfigWithOverrides:
            def __init__(self, base_settings, config_overrides):
                self._base = base_settings
                self._overrides = config_overrides or {}

                # Create nested attribute access for agents, content, reflection, etc.
                self.agents = type("agents", (), {})()
                self.content = type("content", (), {})()
                self.reflection = type("reflection", (), {})()

                # Initialize all configuration attributes
                self._setup_configuration()

            def _has_override(self, key):
                """Check if override exists, handling both dict and object types"""
                if isinstance(self._overrides, dict):
                    return key in self._overrides
                else:
                    return hasattr(self._overrides, key)

            def _get_override(self, key, default=None):
                """Get override value, handling both dict and object types"""
                if isinstance(self._overrides, dict):
                    return self._overrides.get(key, default)
                else:
                    return getattr(self._overrides, key, default)

            def _setup_configuration(self):
                """Setup configuration with overrides"""
                # Set agents configuration with overrides
                if self._has_override("max_search_results"):
                    self.agents.max_search_results = self._get_override(
                        "max_search_results"
                    )
                else:
                    self.agents.max_search_results = getattr(
                        base_settings.agents, "max_search_results", 5
                    )

                if self._has_override("enable_deep_thinking"):
                    self.agents.enable_deep_thinking = self._get_override(
                        "enable_deep_thinking"
                    )
                else:
                    self.agents.enable_deep_thinking = getattr(
                        base_settings.agents, "enable_deep_thinking", False
                    )

                if self._has_override("max_plan_iterations"):
                    self.agents.max_plan_iterations = self._get_override(
                        "max_plan_iterations"
                    )
                else:
                    self.agents.max_plan_iterations = getattr(
                        base_settings.agents, "max_plan_iterations", 3
                    )

                # Set content configuration
                self.content.enable_smart_filtering = getattr(
                    base_settings.content, "enable_smart_filtering", True
                )
                self.content.enable_content_summarization = getattr(
                    base_settings.content, "enable_content_summarization", True
                )
                self.content.summary_type = getattr(
                    base_settings.content, "summary_type", "comprehensive"
                )

                # Pass through other base settings attributes
                self.model_token_limits = getattr(
                    base_settings, "model_token_limits", {}
                )

                # Simplified reflection configuration with overrides - removed redundant top-level assignments

                # Set reflection configuration with override support
                if self._has_override("reflection_model"):
                    self.reflection.model = self._get_override("reflection_model")
                else:
                    reflection_model = getattr(base_settings.reflection, "model", None)
                    if reflection_model is None:
                        # Use reasoning model if enabled, otherwise use basic model
                        if self.agents.enable_deep_thinking:
                            self.reflection.model = "reasoning"
                        else:
                            self.reflection.model = "basic"
                    else:
                        self.reflection.model = reflection_model

                # Removed backward compatibility assignment for reflection_model

                # Set reflection object attributes with simplified configuration and overrides
                reflection_overrides = (
                    self._overrides.get("reflection", {})
                    if isinstance(self._overrides, dict)
                    else {}
                )

                if reflection_overrides.get("enabled") is not None:
                    self.reflection.enabled = reflection_overrides["enabled"]
                else:
                    self.reflection.enabled = getattr(
                        base_settings.reflection, "enabled", True
                    )

                if reflection_overrides.get("max_loops") is not None:
                    self.reflection.max_loops = reflection_overrides["max_loops"]
                else:
                    self.reflection.max_loops = getattr(
                        base_settings.reflection, "max_loops", 2
                    )

                if reflection_overrides.get("quality_threshold") is not None:
                    self.reflection.quality_threshold = reflection_overrides[
                        "quality_threshold"
                    ]
                else:
                    self.reflection.quality_threshold = getattr(
                        base_settings.reflection, "quality_threshold", 0.75
                    )

                # Use unified quality_threshold for both knowledge gap and sufficiency decisions
                quality_threshold = self.reflection.quality_threshold
                self.knowledge_gap_threshold = quality_threshold
                self.sufficiency_threshold = quality_threshold

                # Default behavior: skip initial stage reflection (simplified approach)
                self.reflection.skip_initial_stage_reflection = True
                # Removed backward compatibility assignments for enabled and skip_initial_stage_reflection

                # Max step num configuration with override - prioritize API override
                if self._has_override("max_step_num"):
                    self.max_step_num = self._get_override("max_step_num")
                else:
                    self.max_step_num = getattr(base_settings.agents, "max_step_num", 3)

                # Resources configuration with override
                if self._has_override("resources"):
                    self.resources = self._get_override("resources")
                else:
                    self.resources = getattr(base_settings, "resources", [])

                # MCP configuration
                self.mcp = getattr(
                    base_settings,
                    "mcp",
                    type("mcp", (), {"enabled": False, "servers": [], "timeout": 30})(),
                )

        if isinstance(config, dict) and "configurable" in config:
            config = config["configurable"]
        return ConfigWithOverrides(base_settings, config)

    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        raise RuntimeError(
            f"Configuration loading failed: {e}. Please ensure configuration file is valid and accessible."
        ) from e


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    configurable = get_configuration_from_config(config)
    query = state.get("research_topic")
    background_investigation_results = None

    # get the background investigation results
    from src.llms.error_handler import safe_llm_call

    # Get search engine from settings
    settings = get_settings()
    selected_search_engine = settings.tools.search_engine

    if selected_search_engine == SearchEngine.TAVILY.value:
        searched_content = safe_llm_call(
            LoggedTavilySearch(
                max_results=configurable.agents.max_search_results
            ).invoke,
            query,
            operation_name="Background Investigation - Tavily Search",
            context="Search for background information",
        )
        if isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem['title']}\n\n{elem['content']}" for elem in searched_content
            ]
            raw_results = "\n\n".join(background_investigation_results)
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
            return {"background_investigation_results": ""}
    else:
        background_investigation_results = safe_llm_call(
            get_web_search_tool(
                configurable.agents.max_search_results,
                configurable.content.enable_smart_filtering,
            ).invoke,
            query,
            operation_name="Background Investigation - Web Search",
            context="Search for background information",
        )
        raw_results = json.dumps(background_investigation_results, ensure_ascii=False)

    # Smart content processing is always enabled (smart chunking, summarization, and smart filtering)
    if True:  # Always process content with smart features
        try:
            from src.utils.tokens.content_processor import ContentProcessor

            processor = ContentProcessor(configurable.model_token_limits)

            # get the current llm model name with proper fallback
            model_name = "deepseek-chat"  # Default fallback
            try:
                # Get planner LLM type from settings
                planner_llm_type = getattr(settings.agent_llm_map, "planner", "basic")
                current_llm = get_llm_by_type(planner_llm_type)
                model_name = getattr(
                    current_llm, "model_name", getattr(current_llm, "model", None)
                )
                # If model_name is None or 'unknown', use fallback based on agent type
                if not model_name or model_name == "unknown":
                    model_name = (
                        "deepseek-reasoner"
                        if planner_llm_type == "reasoning"
                        else "deepseek-chat"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to get LLM instance, trying to get model name from config: {e}"
                )
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                try:
                    from src.config.config_loader import load_configuration

                    config = load_configuration()
                    planner_llm_type = getattr(config.agent_llm_map, "planner", "basic")

                    if planner_llm_type == "basic" and config.llm.basic_model:
                        model_name = config.llm.basic_model.model
                    elif planner_llm_type == "reasoning" and config.llm.reasoning_model:
                        model_name = config.llm.reasoning_model.model
                    else:
                        model_name = "deepseek-chat"  # Default fallback

                    logger.debug(
                        f"Retrieved model name from config for planner: {model_name}"
                    )
                    # Create a minimal LLM instance for processing
                    current_llm = get_llm_by_type("basic")  # Fallback to basic model
                except Exception as config_e:
                    logger.error(f"Failed to get model name from config: {config_e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    model_name = "deepseek-chat"  # Final fallback
                    current_llm = get_llm_by_type("basic")  # Fallback to basic model

            # Preemptive token check for raw search results
            model_limits = processor.get_model_limits(model_name)
            results_within_limit, results_token_result = (
                processor.check_content_token_limit(
                    raw_results,
                    model_name,
                    model_limits.safe_input_limit,
                    model_limits.safety_margin,
                )
            )
            token_limit_exceeded = not results_within_limit

            # Check if token limit is exceeded and apply fallback processing if needed
            if token_limit_exceeded:
                logger.info(
                    f"Applying fallback processing to search results "
                    f"(tokens: {results_token_result.total_tokens}, limit: {model_limits.safe_input_limit})"
                )

                # Use traditional processing as fallback
                if configurable.content.enable_content_summarization:
                    # Use smart summarization
                    processed_results = processor.summarize_content(
                        raw_results,
                        current_llm,
                        model_name,
                        configurable.content.summary_type,
                    )
                    logger.info(
                        f"Search results summarized: {len(raw_results)} -> {len(processed_results)} characters"
                    )
                else:
                    # Use smart chunking (always enabled with auto strategy)
                    chunks = processor.smart_chunk_content(
                        raw_results, model_name, "auto"
                    )
                    processed_results = chunks[0] if chunks else raw_results
                    logger.info(
                        f"Search results chunked: {len(raw_results)} -> {len(processed_results)} characters"
                    )

                return {"background_investigation_results": processed_results}
            else:
                logger.info("Search results within token limit, no processing needed")

        except Exception as e:
            logger.error(
                f"Smart content processing failed: {e}, using original results"
            )
            logger.error(f"Full traceback: {traceback.format_exc()}")

    return {"background_investigation_results": raw_results}


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan with Phase 5 reflection integration."""
    logger.info("Planner generating full plan with enhanced reflection integration")
    configurable = get_configuration_from_config(config)

    # Phase 1 Simplification: Initialize unified config for planner
    from src.utils.researcher.isolation_config_manager import IsolationConfigManager

    config_manager = IsolationConfigManager(configurable)
    unified_config = config_manager.get_unified_config()

    # Debug: Log the max_step_num value from unified config
    max_step_num_value = unified_config.max_step_num
    logger.info(f"DEBUG: unified_config.max_step_num = {max_step_num_value}")

    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    messages = apply_prompt_template("planner", state, configurable)

    # Phase 1 Simplification: Use unified config for reflection
    reflection_config = {
        "reflection_enabled": getattr(unified_config, "enabled", True),
        "reflection_max_loops": getattr(unified_config, "max_loops", 1),
        "reflection_model": getattr(unified_config, "model", None),
        "reflection_quality_threshold": getattr(
            unified_config, "quality_threshold", 0.7
        ),
    }

    # Phase 5: Initialize reflection agent for planner if enabled
    reflection_agent = None
    if reflection_config.get("reflection_enabled", True):
        try:
            from src.utils.reflection.enhanced_reflection import (
                EnhancedReflectionAgent,
                # ReflectionConfig removed - using unified ReflectionSettings from src.config.models
            )

            # ReflectionConfig removed - using unified config system
            # reflection_agent_config = {...}

            reflection_agent = EnhancedReflectionAgent(config=configurable)

            logger.info("Enhanced reflection agent initialized for planner")

        except Exception as e:
            logger.error(f"Failed to initialize reflection agent for planner: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            reflection_agent = None

    # Phase 5: Check for reflection insights from previous research
    reflection_insights = None
    if hasattr(state, "get") and state.get("observations"):
        for observation in state.get("observations", []):
            if isinstance(observation, dict) and "reflection_insights" in observation:
                reflection_insights = observation["reflection_insights"]
                break

    # Phase 5: Integrate reflection-driven planning adjustments with enhanced analysis
    reflection_context = ""
    if reflection_insights and reflection_config.get("reflection_enabled", True):
        logger.info("Integrating reflection insights into planning")

        # Add reflection context to planning messages
        reflection_context = f"""
        
REFLECTION INSIGHTS FROM PREVIOUS RESEARCH:
        - Knowledge Gaps Identified: {1 if getattr(reflection_insights, 'primary_knowledge_gap', None) else 0}
        - Research Sufficiency: {'Sufficient' if getattr(reflection_insights, 'is_sufficient', False) else 'Insufficient'}
        - Suggested Follow-up Query: {getattr(reflection_insights, 'primary_follow_up_query', 'None')}
        - Confidence Score: {getattr(reflection_insights, 'confidence_score', 'N/A')}
        
        Please adjust the research plan to address these knowledge gaps and incorporate the suggested follow-up queries.
        """

        # Phase 5: Perform planner-specific reflection on the planning context
        if reflection_agent is not None:
            try:
                logger.info("Performing planner-specific reflection analysis")

                # Analyze the planning context for completeness
                {
                    "current_plan_context": str(state.get("current_plan", ""))[:1000],
                    "reflection_insights": reflection_insights,
                    "planning_objective": state.get("research_topic", ""),
                    "available_information": state.get("context", {}),
                    "plan_iterations": plan_iterations,
                }

                # Note: planner_node is not async, so we use sync call or skip reflection
                # For now, we'll skip the planner-specific reflection to avoid async issues
                # planner_reflection = await reflection_agent.analyze_knowledge_gaps(planning_context)
                logger.info(
                    "Skipping planner-specific reflection (async not supported in sync function)"
                )
                planner_reflection = None

                # Add planner reflection to insights
                if planner_reflection and planner_reflection.primary_knowledge_gap:
                    reflection_context += f"""

## Planner Reflection Analysis

**Planning Completeness**: {"Complete" if planner_reflection.is_sufficient else "Incomplete"}
**Planning Confidence**: {planner_reflection.confidence:.2f}

**Planning Gaps Identified**:
{
                        f"- {planner_reflection.primary_follow_up_query}" if planner_reflection.primary_follow_up_query else "- No planning gaps identified"
                    }
                    """.strip()

                    logger.info(
                        f"Planner reflection completed - Sufficient: {planner_reflection.is_sufficient}, Gaps: {1 if planner_reflection.primary_knowledge_gap else 0}"
                    )
                elif planner_reflection is None:
                    logger.info(
                        "Planner reflection was skipped due to sync function limitations"
                    )

            except Exception as e:
                logger.error(f"Planner reflection failed: {e}")
                logger.warning("Proceeding without planner-specific reflection")

        if reflection_context.strip():
            messages.append({"role": "user", "content": reflection_context})

    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        # Preemptive token check before adding background investigation results
        background_content = (
            "background investigation results of user query:\n"
            + state["background_investigation_results"]
            + "\n"
        )

        # Check if adding background results would exceed token limits
        try:
            from src.utils.tokens.content_processor import ContentProcessor

            processor = ContentProcessor(configurable.model_token_limits)

            # Get current model name
            model_name = "deepseek-chat"  # Default fallback
            try:
                from src.llms.llm import get_llm_by_type

                settings = get_settings()
                planner_llm_type = getattr(settings.agent_llm_map, "planner", "basic")
                current_llm = get_llm_by_type(planner_llm_type)
                model_name = getattr(
                    current_llm, "model_name", getattr(current_llm, "model", None)
                )
                if not model_name or model_name == "unknown":
                    model_name = (
                        "deepseek-reasoner"
                        if planner_llm_type == "reasoning"
                        else "deepseek-chat"
                    )
            except Exception:
                pass

            # Calculate combined token count for messages + background content
            messages_text = json.dumps(messages, ensure_ascii=False)
            combined_content = messages_text + background_content

            model_limits = processor.get_model_limits(model_name)
            content_within_limit, token_result = processor.check_content_token_limit(
                combined_content,
                model_name,
                model_limits.safe_input_limit,
                model_limits.safety_margin,
            )

            if content_within_limit:
                messages += [
                    {
                        "role": "user",
                        "content": background_content,
                    }
                ]
                logger.info(
                    f"Added background investigation results to planner context "
                    f"({token_result.total_tokens} tokens, within limit)"
                )
            else:
                # Summarize background results if they would cause overflow
                logger.warning(
                    f"Background investigation results too large ({token_result.total_tokens} tokens), "
                    f"summarizing to fit within {model_limits.safe_input_limit} token limit"
                )

                from src.llms.llm import get_llm_by_type

                summarized_background = processor.summarize_content(
                    state["background_investigation_results"],
                    get_llm_by_type("basic"),
                    model_name,
                    "key_points",
                )

                messages += [
                    {
                        "role": "user",
                        "content": (
                            "background investigation results of user query (summarized):\n"
                            + summarized_background
                            + "\n"
                        ),
                    }
                ]
                logger.info(
                    "Added summarized background investigation results to planner context"
                )

        except Exception as e:
            logger.error(
                f"Failed to check token limits for background investigation: {e}"
            )
            # Fallback to original behavior
            messages += [
                {
                    "role": "user",
                    "content": background_content,
                }
            ]

    # Configure LLM based on settings
    from src.llms.llm import get_llm_by_type

    settings = get_settings()
    planner_llm_type = getattr(settings.agent_llm_map, "planner", "basic")
    use_structured_output = (
        planner_llm_type == "basic" and not configurable.agents.enable_deep_thinking
    )

    if configurable.agents.enable_deep_thinking:
        llm = get_llm_by_type("reasoning")
        # llm = get_llm_by_type("reasoning").with_structured_output(
        #    Plan,
        #    method="transform",
        #    )
    elif use_structured_output:
        llm = get_llm_by_type("basic").with_structured_output(
            Plan,
            method="json_mode",
        )
    else:
        llm = get_llm_by_type(planner_llm_type)

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.agents.max_plan_iterations:
        return Command(
            update={
                # Preserve research_topic and resources from coordinator_node
                "research_topic": state.get("research_topic", ""),
                "resources": state.get("resources", []),
            },
            goto="context_optimizer",
        )

    # Context evaluation will be handled automatically by safe_llm_call

    full_response = ""
    if use_structured_output:
        response = safe_llm_call(
            llm.invoke,
            messages,
            operation_name="Planner",
            context="Generate the full plan.",
        )
        if hasattr(response, "model_dump_json"):
            full_response = response.model_dump_json(indent=4, exclude_none=True)
        else:
            # if the response is not a structured output, return the default plan
            full_response = '{"steps": [{"title": "Research Task", "description": "Continue with general research due to content limitations."}]}'
    else:

        def stream_llm():
            response = safe_llm_call(
                llm.stream,
                messages,
                operation_name="LLM streaming call",
                context="Stream LLM response",
            )
            content = ""
            for chunk in response:
                content += chunk.content
            return type("StreamResponse", (), {"content": content})()

        response = safe_llm_call(
            stream_llm,
            operation_name="Planner streaming call",
            context="Generate research plan",
        )
        full_response = (
            response.content if hasattr(response, "content") else str(response)
        )
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError as e:
        logger.warning(f"Planner response is not a valid JSON: {e}")
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        if plan_iterations > 0:
            return Command(
                update={
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="context_optimizer",
            )
        else:
            return Command(
                update={
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="__end__",
            )
    if curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        try:
            new_plan = Plan.model_validate(curr_plan)
        except Exception as e:
            logger.warning(
                f"Planner execution error: Failed to parse Plan from completion {curr_plan}. Got: {e}"
            )
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            if plan_iterations > 0:
                return Command(
                    update={
                        "research_topic": state.get("research_topic", ""),
                        "resources": state.get("resources", []),
                        "locale": state.get("locale", "en-US"),
                    },
                    goto="context_optimizer",
                )
            else:
                return Command(
                    update={
                        "research_topic": state.get("research_topic", ""),
                        "resources": state.get("resources", []),
                        "locale": state.get("locale", "en-US"),
                    },
                    goto="__end__",
                )
        # Phase 5: Add reflection metadata to command result
        reflection_metadata = {
            "reflection_applied": (
                reflection_config.get("enabled", True)
                and reflection_insights is not None
            ),
            "timestamp": datetime.now().isoformat(),
            "planner_reflection_enabled": reflection_agent is not None,
            "reflection_config": reflection_config,
        }

        if reflection_insights:
            reflection_metadata.update(
                {
                    "knowledge_gaps_count": (
                        1
                        if getattr(reflection_insights, "primary_knowledge_gap", None)
                        else 0
                    ),
                    "confidence": getattr(reflection_insights, "confidence_score", 0.0),
                    "is_sufficient": getattr(
                        reflection_insights, "is_sufficient", False
                    ),
                }
            )

        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
                "reflection_metadata": reflection_metadata,
                "plan_iterations": plan_iterations + 1,
                "research_topic": state.get("research_topic", ""),
                "resources": state.get("resources", []),
                "locale": state.get("locale", "en-US"),
            },
            goto="context_optimizer",
        )

    # Phase 5: Add reflection metadata for human feedback path as well
    reflection_metadata = {
        "reflection_applied": (
            reflection_config.get("enabled", True) and reflection_insights is not None
        ),
        "timestamp": datetime.now().isoformat(),
        "planner_reflection_enabled": reflection_agent is not None,
        "reflection_config": reflection_config,
    }

    if reflection_insights:
        reflection_metadata.update(
            {
                "knowledge_gaps_count": (
                    1
                    if getattr(reflection_insights, "primary_knowledge_gap", None)
                    else 0
                ),
                "confidence": getattr(reflection_insights, "confidence_score", 0.0),
                "is_sufficient": getattr(reflection_insights, "is_sufficient", False),
            }
        )

    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
            "reflection_metadata": reflection_metadata,
            "plan_iterations": plan_iterations + 1,
            "research_topic": state.get("research_topic", ""),
            "resources": state.get("resources", []),
            "locale": state.get("locale", "en-US"),
        },
        goto="human_feedback",
    )


def human_feedback_node(
    state,
) -> Command[Literal["planner", "research_team", "reporter", "__end__"]]:
    current_plan = state.get("current_plan", "")
    # check if the plan is auto accepted
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    if not auto_accepted_plan:
        feedback = interrupt("Please Review the Plan.")

        # if the feedback is not accepted, return the planner node
        if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="planner",
            )
        elif feedback and str(feedback).upper().startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            raise TypeError(f"Interrupt value of {feedback} is not supported.")

    # if the plan is accepted, run the following node
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    goto = "research_team"
    try:
        current_plan = repair_json_output(current_plan)
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(current_plan)
        if new_plan["has_enough_context"]:
            goto = "context_optimizer"
    except json.JSONDecodeError as e:
        logger.warning(f"Planner response is not a valid JSON: {e}")
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        if plan_iterations > 1:  # the plan_iterations is increased before this check
            return Command(
                update={
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="context_optimizer",
            )
        else:
            return Command(
                update={
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="__end__",
            )

    try:
        validated_plan = Plan.model_validate(new_plan)
    except Exception as e:
        logger.warning(
            f"Human feedback node execution error: Failed to parse Plan from completion {new_plan}. Got: {e}"
        )
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        if plan_iterations > 1:
            return Command(
                update={
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="context_optimizer",
            )
        else:
            return Command(
                update={
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                    "locale": state.get("locale", "en-US"),
                },
                goto="__end__",
            )

    return Command(
        update={
            "current_plan": validated_plan,
            "plan_iterations": plan_iterations,
            "locale": new_plan["locale"],
            # Preserve research_topic and resources from coordinator_node
            "research_topic": state.get("research_topic", ""),
            "resources": state.get("resources", []),
        },
        goto=goto,
    )


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    configurable = get_configuration_from_config(config)
    messages = apply_prompt_template("coordinator", state, configurable)

    # Context evaluation will be handled automatically by safe_llm_call
    settings = get_settings()
    coordinator_llm_type = getattr(settings.agent_llm_map, "coordinator", "basic")
    llm = get_llm_by_type(coordinator_llm_type).bind_tools([handoff_to_planner])

    response = safe_llm_call(
        llm.invoke,
        messages,
        operation_name="Coordinator",
        context="Coordinating with customers",
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    locale = state.get("locale", "en-US")  # Default locale if not specified
    research_topic = state.get("research_topic", "")

    if len(response.tool_calls) > 0:
        goto = "planner"
        if state.get("enable_background_investigation"):
            # if the search_before_planning is True, add the web search tool to the planner agent
            goto = "background_investigator"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_call.get("args", {}).get("locale") and tool_call.get(
                    "args", {}
                ).get("research_topic"):
                    locale = tool_call.get("args", {}).get("locale")
                    research_topic = tool_call.get("args", {}).get("research_topic")
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    else:
        logger.warning(
            "Coordinator response contains no tool calls. Terminating workflow execution."
        )
        logger.debug(f"Coordinator response: {response}")

    return Command(
        update={
            "locale": locale,
            "research_topic": research_topic,
            "resources": configurable.resources,
        },
        goto=goto,
    )


def reporter_node(state: State, config: RunnableConfig):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    configurable = get_configuration_from_config(config)
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    invoke_messages = apply_prompt_template("reporter", input_, configurable)
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
            name="system",
        )
    )

    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")

    # Context evaluation will be handled automatically by safe_llm_call
    settings = get_settings()
    reporter_llm_type = getattr(settings.agent_llm_map, "reporter", "basic")
    llm = get_llm_by_type(reporter_llm_type)

    response = safe_llm_call(
        llm.invoke,
        invoke_messages,
        operation_name="Reporter",
        context="Generate the final report",
    )
    response_content = (
        response.content if hasattr(response, "content") else str(response)
    )
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}


async def research_team_node(state: State, config: RunnableConfig):
    """Research team node that collaborates on tasks with parallel execution support."""
    logger.info("Research team is collaborating on tasks.")

    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        logger.warning("No plan or steps found")
        return Command(goto="planner")

    # Check if all steps are completed
    if all(step.execution_res for step in current_plan.steps):
        logger.info("All steps completed, moving to context_optimizer")
        return Command(goto="context_optimizer")

    # Phase 1 Simplification: Use unified config for parallel execution
    configurable = get_configuration_from_config(config)
    from src.utils.researcher.isolation_config_manager import IsolationConfigManager

    config_manager = IsolationConfigManager(configurable)
    unified_config = config_manager.get_unified_config()

    enable_parallel = unified_config.enable_parallel_execution
    max_parallel_tasks = unified_config.max_parallel_tasks

    if enable_parallel:
        return await _execute_steps_parallel(state, config, max_parallel_tasks)
    else:
        # Fallback to serial execution
        return await _execute_steps_serial(state, config)


def _analyze_step_dependencies(steps: List[Step]) -> Dict[str, Set[str]]:
    """Analyze dependencies between steps based on their descriptions.

    Returns a dictionary where keys are step titles and values are sets of prerequisite step titles.
    """
    dependencies = {}

    for step in steps:
        step_deps = set()
        step_title_lower = step.title.lower()
        step_desc_lower = step.description.lower()

        # Check if this step references other steps
        for other_step in steps:
            if other_step.title == step.title:
                continue

            other_title_lower = other_step.title.lower()

            # Simple dependency detection based on keywords
            dependency_keywords = [
                "based on",
                "using",
                "from",
                "after",
                "following",
                "according to",
                "based on",
                "using",
                "according to",
                "after",
                "reference",
            ]

            # Check if current step description mentions other step
            if any(keyword in step_desc_lower for keyword in dependency_keywords):
                if other_title_lower in step_desc_lower:
                    step_deps.add(other_step.title)

            # Processing steps typically depend on research steps
            if (
                step.step_type == StepType.PROCESSING
                and other_step.step_type == StepType.RESEARCH
            ):
                # Check for topic overlap
                step_words = set(step_title_lower.split())
                other_words = set(other_title_lower.split())
                if len(step_words.intersection(other_words)) >= 2:
                    step_deps.add(other_step.title)

        dependencies[step.title] = step_deps

    return dependencies


def _get_executable_steps(
    steps: List[Step], dependencies: Dict[str, Set[str]]
) -> List[Step]:
    """Get steps that can be executed now (no unmet dependencies)."""
    executable = []

    for step in steps:
        if step.execution_res:  # Already completed
            continue

        # Check if all dependencies are satisfied
        step_deps = dependencies.get(step.title, set())
        deps_satisfied = True

        for dep_title in step_deps:
            dep_step = next((s for s in steps if s.title == dep_title), None)
            if not dep_step or not dep_step.execution_res:
                deps_satisfied = False
                break

        if deps_satisfied:
            executable.append(step)

    return executable


async def _execute_steps_parallel(
    state: State, config: RunnableConfig, max_parallel_tasks: int
) -> Command:
    """
    Executes research steps in parallel, correctly handling and merging state updates from each step,
    especially dynamically added steps from the reflection process.
    """
    logger.info(
        f"Starting parallel execution with max {max_parallel_tasks} concurrent tasks."
    )

    from src.utils.performance.parallel_executor import (
        create_parallel_executor,
        ParallelTask,
        TaskPriority,
    )

    current_plan = state.get("current_plan")
    executable_steps = [step for step in current_plan.steps if not step.execution_res]

    if not executable_steps:
        logger.info("No executable steps found, proceeding to planner.")
        return Command(goto="planner")

    # Prepare tasks for parallel execution
    parallel_tasks = []
    for step in executable_steps:
        agent_type = "coder" if step.step_type == StepType.PROCESSING else "researcher"
        task = ParallelTask(
            task_id=step.title,
            func=_execute_single_step_parallel,
            args=(state, config, step, agent_type),
            priority=TaskPriority.NORMAL,
        )
        parallel_tasks.append(task)

    # Execute tasks and process results
    executor = create_parallel_executor(max_concurrent_tasks=max_parallel_tasks)
    executor.add_tasks(parallel_tasks)
    results = await executor.execute_all()

    # --- State Synchronization Logic ---
    # Start with the original plan and observations
    final_plan = current_plan.copy()
    final_observations = list(state.get("observations", []))
    all_messages = list(state.get("messages", []))
    newly_added_steps = []

    for task_id, task_result in results.items():
        if task_result.status.value == "completed":
            result_data = task_result.result
            original_step_title = result_data.get("original_step_title")

            # Find the step in our final_plan to update its execution result
            step_to_update = next(
                (s for s in final_plan.steps if s.title == original_step_title), None
            )
            if not step_to_update:
                continue

            # Update the execution result of the original step
            if result_data.get("status") != "failed":
                updated_plan = result_data.get("updated_plan")
                if updated_plan and updated_plan.steps:
                    updated_step_in_plan = updated_plan.steps[0]
                    step_to_update.execution_res = updated_step_in_plan.execution_res
                    all_messages.append(
                        AIMessage(
                            content=step_to_update.execution_res, name="researcher"
                        )
                    )
                else:
                    step_to_update.execution_res = (
                        "Execution completed but no result available"
                    )
            else:
                step_to_update.execution_res = (
                    f"Execution failed: {result_data.get('error')}"
                )

            # Add any new observations
            final_observations.extend(result_data.get("new_observations", []))

            # If the step generated a new plan, collect the *new* steps
            if result_data.get("status") == "completed_with_update":
                extended_plan = result_data.get("updated_plan")
                if extended_plan and extended_plan.steps:
                    # Identify only the steps that are genuinely new
                    original_titles = {s.title for s in current_plan.steps}
                    for step in extended_plan.steps:
                        if step.title not in original_titles:
                            newly_added_steps.append(step)
                            logger.info(
                                f"Collected new step '{step.title}' from reflection."
                            )
                else:
                    logger.warning(
                        "Step marked as 'completed_with_update' but no valid updated_plan found"
                    )

    # Merge newly added steps into the final plan
    if newly_added_steps:
        final_plan.steps.extend(newly_added_steps)
        logger.info(f"Added {len(newly_added_steps)} new steps to the plan.")

    # --- Final Decision Logic ---
    # Decide the next action based on the fully synchronized and updated plan
    if all(step.execution_res for step in final_plan.steps):
        logger.info(
            "All steps in the synchronized plan are completed. Moving to planner."
        )
        next_node = "planner"
    else:
        remaining_steps = [s.title for s in final_plan.steps if not s.execution_res]
        logger.info(
            f"There are still {len(remaining_steps)} uncompleted steps: {remaining_steps}. Continuing research."
        )
        next_node = "research_team"

    return Command(
        goto=next_node,
        update={
            "current_plan": final_plan,
            "observations": final_observations,
            "messages": all_messages,
        },
    )


async def _execute_single_step_parallel(
    state: State, config: RunnableConfig, step: Step, agent_type: str
) -> Dict[str, Any]:
    """
    Executes a single step in a parallel context, ensuring that any state updates
    (like plan modifications from reflection) are properly captured and returned in a structured format.
    """
    logger.info(f"Executing step in parallel: {step.title} with {agent_type}")

    # Create a temporary, isolated state for this specific step execution.
    # This prevents concurrent steps from interfering with each other.
    current_plan = state.get("current_plan")
    temp_state = state.copy()
    # The plan for the sub-task only contains the single step to be executed.
    temp_state["current_plan"] = current_plan.copy(update={"steps": [step]})

    try:
        # Route to the appropriate node based on the agent type.
        if agent_type == "researcher":
            result_command = await researcher_node(temp_state, config)
        elif agent_type == "coder":
            result_command = await coder_node(temp_state, config)
        else:
            result_command = await researcher_node(temp_state, config)

        # Process the result, which is expected to be a Command object.
        if isinstance(result_command, Command) and result_command.update:
            updated_plan = result_command.update.get("current_plan")
            new_observations = result_command.update.get("observations", [])

            if updated_plan:
                # Find the result of the specific step that was executed.
                executed_step = next(
                    (s for s in updated_plan.steps if s.title == step.title), None
                )
                if executed_step and executed_step.execution_res:
                    plan_was_extended = len(updated_plan.steps) > 1
                    logger.info(f"Step '{step.title}' completed successfully.")
                    return {
                        "status": (
                            "completed_with_update"
                            if plan_was_extended
                            else "completed"
                        ),
                        "original_step_title": step.title,
                        "updated_plan": updated_plan,
                        "new_observations": new_observations,
                    }

        # This block is reached if the result was not a valid Command or the step result was missing.
        logger.warning(
            f"Could not extract proper result for step '{step.title}', using fallback."
        )
        return {
            "status": "failed",
            "original_step_title": step.title,
            "error": "Result extraction failed.",
        }

    except Exception as e:
        logger.error(
            f"Error executing step '{step.title}' in parallel: {e}", exc_info=True
        )
        return {
            "status": "failed",
            "original_step_title": step.title,
            "error": str(e),
        }


async def _execute_steps_serial(state: State, config: RunnableConfig) -> Command:
    """Fallback to serial execution (original behavior)."""
    logger.info("Using serial execution mode")

    current_plan = state.get("current_plan")

    # Find the first unexecuted step
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break

    if not current_step:
        return Command(goto="planner")

    # Execute single step based on type
    if current_step.step_type == StepType.RESEARCH:
        return await researcher_node(state, config)
    elif current_step.step_type == StepType.PROCESSING:
        return await coder_node(state, config)
    else:
        return await researcher_node(state, config)


async def _execute_agent_step(
    state: State, config: RunnableConfig, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent.

    Now uses unified ExecutionContextManager for consistent context handling.
    """
    from src.utils.context.execution_context_manager import (
        ExecutionContextManager,
        ContextConfig,
    )

    current_plan = state.get("current_plan")
    observations = state.get("observations", [])

    # Find the first unexecuted step
    current_step = None
    completed_steps = []
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
        else:
            completed_steps.append(
                {
                    "step": step.title,
                    "description": step.description,
                    "execution_res": step.execution_res,
                }
            )

    if not current_step:
        logger.warning("No unexecuted step found")
        return Command(goto="research_team")

    logger.info(f"Executing step: {current_step.title}, agent: {agent_name}")

    # Phase 1 Simplification: Use unified config for context management
    configurable = get_configuration_from_config(config)
    from src.utils.researcher.isolation_config_manager import IsolationConfigManager

    config_manager = IsolationConfigManager(configurable)
    unified_config = config_manager.get_unified_config()

    context_config = ContextConfig(
        max_context_steps=unified_config.max_context_steps_parallel,
        max_step_content_length=2000,
        max_observations_length=10000,
        enable_content_deduplication=True,
        enable_smart_truncation=True,
    )

    # Check if context sharing should be disabled
    if unified_config.disable_context_parallel:
        context_config.max_context_steps = 0
        logger.info("Context sharing disabled - no historical context will be included")

    context_manager = ExecutionContextManager(context_config)

    # Prepare optimized context using unified manager
    current_step_dict = {
        "step": current_step.title,
        "description": current_step.description,
    }

    optimized_steps, context_info = context_manager.prepare_context_for_execution(
        completed_steps, current_step_dict, agent_name
    )

    # Prepare the agent state with current task information
    # This allows the agent's prompt system to properly render the template
    agent_state = {
        **state,  # Include all existing state
        "current_step_title": current_step.title,
        "current_step_description": current_step.description,
        "context_info": context_info,
        "completed_steps": optimized_steps,
        "agent_name": agent_name,
        "original_user_query": state.get(
            "research_topic", ""
        ),  # Add original user query for context
    }

    agent_state["messages"] = [HumanMessage(content=current_step.description)]

    # Add resources information for researcher agent
    if agent_name == "researcher" and state.get("resources"):
        resources_info = "**The user mentioned the following resource files:**\n\n"
        for resource in state.get("resources"):
            resources_info += f"- {resource.title} ({resource.description})\n"
        resources_info += "\n\nYou MUST use the **local_search_tool** to retrieve the information from the resource files."
        agent_state["resources_info"] = resources_info

        # Add citation reminder
        agent_state["citation_reminder"] = (
            "IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)"
        )

    # Invoke the agent
    default_recursion_limit = 10
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        recursion_limit = default_recursion_limit

    logger.info(
        f"Agent state prepared for {agent_name} with step: {current_step.title}"
    )

    # Context evaluation will be handled automatically by safe_llm_call_async
    # The agent's prompt system will handle template rendering with the agent_state

    result = await safe_llm_call_async(
        agent.ainvoke,
        input=agent_state,
        config={"recursion_limit": recursion_limit},
        operation_name=f"{agent_name} executor",
        context=f"Execute step: {current_step.title}",
        enable_smart_processing=True,
        max_retries=3,
    )

    # Process the result
    if hasattr(result, "get") and "messages" in result:
        response_content = result["messages"][-1].content
    else:
        # If the result is not a message, use the content directly
        response_content = result.content if hasattr(result, "content") else str(result)

    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Use advanced observation management
    new_observation = (
        f"Step: {current_step.title}\nAgent: {agent_name}\nResult: {response_content}"
    )
    optimized_observations = context_manager.manage_observations_advanced(
        observations + [new_observation], optimization_level="standard"
    )

    # Create updated step with execution result (immutable pattern)
    updated_step = current_step.copy(update={"execution_res": response_content})

    # Create updated plan with the new step
    updated_steps = list(current_plan.steps)
    step_index = next(
        (i for i, s in enumerate(current_plan.steps) if s.title == current_step.title),
        None,
    )
    if step_index is not None:
        updated_steps[step_index] = updated_step

    updated_plan = current_plan.copy(update={"steps": updated_steps})
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": optimized_observations,
            "current_plan": updated_plan,
            # Preserve research_topic and resources from coordinator_node
            "research_topic": state.get("research_topic", ""),
            "resources": state.get("resources", []),
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic for both researcher_node and coder_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_type: The type of agent ("researcher" or "coder")
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = get_configuration_from_config(config)
    mcp_servers = {}
    enabled_tools = {}

    # Extract MCP server configuration for this agent type
    if configurable.mcp.enabled and configurable.mcp.servers:
        for server_name, server_config in enumerate(configurable.mcp.servers):
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                server_key = f"server_{server_name}"
                mcp_servers[server_key] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_key

    # Create and execute agent with MCP tools if available
    if configurable.mcp.enabled and mcp_servers:
        async with MultiServerMCPClient(mcp_servers) as client:
            loaded_tools = default_tools[:]
            for tool in client.get_tools():
                if tool.name in enabled_tools:
                    tool.description = (
                        f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                    )
                    loaded_tools.append(tool)
            # Delayed import to avoid circular import
            from src.agents import create_agent

            agent = create_agent(
                agent_type, agent_type, loaded_tools, agent_type, configurable
            )
            return await _execute_agent_step(state, config, agent, agent_type)
    else:
        # Use default tools if no MCP servers are configured
        # Delayed import to avoid circular import
        from src.agents import create_agent

        agent = create_agent(
            agent_type, agent_type, default_tools, agent_type, configurable
        )
        return await _execute_agent_step(state, config, agent, agent_type)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team", "planner"]]:
    """
    Orchestrates the research workflow.
    """
    logger.info("Executing research workflow.")
    try:
        # Create a copy of the state to avoid side effects during the process
        current_state = state.copy()

        # Step 1: Prepare research step
        prepare_result = await prepare_research_step_node(current_state, config)
        current_state.update(prepare_result)

        # Step 2: Execute research agent
        agent_result = await researcher_agent_node(current_state, config)
        current_state.update(agent_result)

        # Step 3: Reflection
        reflection_result = await reflection_node(current_state, config)
        current_state.update(reflection_result)

        # Step 4: Update plan based on research and reflection
        plan_update_result = await update_plan_node(current_state, config)
        current_state.update(plan_update_result)

        # **CRITICAL FIX: Set execution_res for the current step**
        # This ensures that _execute_single_step_parallel can find the executed step
        current_plan = current_state.get("current_plan")
        current_step = current_state.get("current_step")
        research_result = current_state.get("research_result")

        if current_plan and current_step and research_result:
            # Find and update the current step with execution result
            updated_steps = []
            for step in current_plan.steps:
                if step.title == current_step.title:
                    # Create updated step with execution result
                    updated_step = step.model_copy(
                        update={"execution_res": research_result, "completed": True}
                    )
                    updated_steps.append(updated_step)
                    logger.info(f"Set execution_res for step '{step.title}'")
                else:
                    updated_steps.append(step)

            # Update the plan with the modified steps
            updated_plan = current_plan.model_copy(update={"steps": updated_steps})
            current_state["current_plan"] = updated_plan

        # Step 5: Check if the research is complete
        completion_status = check_research_completion_node(current_state)

        # **THE CRITICAL FIX IS HERE**
        # Return a Command object that contains the *full*, potentially modified state.
        # This ensures that any updates to the plan (e.g., new steps from reflection)
        # are propagated back to the main graph state.
        return Command(
            goto=completion_status,
            update={
                "current_plan": current_state.get("current_plan"),
                "observations": current_state.get("observations"),
                "reflection": current_state.get("reflection"),
                # Preserve other essential state keys from the original state
                "research_topic": state.get("research_topic", ""),
                "resources": state.get("resources", []),
                "messages": state.get("messages", []),
            },
        )

    except Exception as e:
        logger.error(f"Research workflow failed: {e}", exc_info=True)
        raise


async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    logger.info("Coder node is coding.")
    return await _setup_and_execute_agent_step(
        state,
        config,
        "coder",
        [python_repl_tool],
    )


async def context_optimizer_node(
    state: State, config: RunnableConfig
) -> Command[Literal["reporter"]]:
    """Context optimization node that compresses observations before report generation.

    This node uses ExecutionContextManager to compress accumulated research summaries
    from parallel researcher nodes to prevent token accumulation issues in the
    reporter node.
    """
    logger.info("Context optimizer node is optimizing observations.")

    observations = state.get("observations", [])

    if not observations:
        logger.info("No observations to optimize, proceeding to reporter.")
        return Command(goto="reporter")

    logger.info(f"Optimizing {len(observations)} observations for context compression.")

    try:
        from src.utils.context.execution_context_manager import (
            ExecutionContextManager,
            ContextConfig,
        )

        # Initialize unified context manager with reporter-specific configuration
        context_config = ContextConfig(
            max_context_steps=10,
            max_step_content_length=1500,
            max_observations_length=5000,
            enable_content_deduplication=True,
            enable_smart_truncation=True,
        )

        context_manager = ExecutionContextManager(context_config)

        # Use advanced observation management for compression
        optimized_observations = context_manager.manage_observations_advanced(
            observations,
            optimization_level="aggressive",  # Use aggressive optimization for reporter
        )

        # Log optimization results
        original_length = sum(len(str(obs)) for obs in observations)
        optimized_length = sum(len(str(obs)) for obs in optimized_observations)
        compression_ratio = (
            (original_length - optimized_length) / original_length
            if original_length > 0
            else 0
        )

        logger.info(
            f"Context optimization completed: "
            f"Original: {len(observations)} observations ({original_length} chars), "
            f"Optimized: {len(optimized_observations)} observations ({optimized_length} chars), "
            f"Compression: {compression_ratio:.2%}"
        )

        return Command(
            update={
                "observations": optimized_observations,
                # Preserve research_topic and resources from coordinator_node
                "research_topic": state.get("research_topic", ""),
                "resources": state.get("resources", []),
            },
            goto="reporter",
        )

    except Exception as e:
        logger.error(f"Context optimization failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.warning(
            "Proceeding with original observations due to optimization failure."
        )

        # Fallback: proceed with original observations if optimization fails
        return Command(
            update={
                # Preserve research_topic and resources from coordinator_node
                "research_topic": state.get("research_topic", ""),
                "resources": state.get("resources", []),
            },
            goto="reporter",
        )


async def planning_context_optimizer_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner"]]:
    """
    Planning context optimizer node - Optimize message history and observation records before re-planning

    Uses ExecutionContextManager for unified context optimization:
    1. Preserve key messages (user requests, latest feedback)
    2. Optimize planning context based on iteration count
    3. Intelligently compress observations
    4. Maintain context coherence
    """
    logger.info("Planning context optimizer started")

    try:
        from src.utils.context.execution_context_manager import (
            ExecutionContextManager,
            ContextConfig,
        )

        messages = state.get("messages", [])
        observations = state.get("observations", [])
        plan_iterations = state.get("plan_iterations", 0)

        # Initialize unified context manager with planning-specific configuration
        context_config = ContextConfig(
            max_context_steps=15,
            max_step_content_length=2000,
            max_observations_length=8000,
            enable_content_deduplication=True,
            enable_smart_truncation=True,
        )

        context_manager = ExecutionContextManager(context_config)
        update_dict = {}

        # Optimize message history using unified manager
        if len(messages) <= 8:
            logger.info(
                f"Message count ({len(messages)}) is low, skipping message optimization"
            )
        else:
            # Use planning context optimization
            optimized_messages, _ = context_manager.optimize_planning_context(
                messages, observations, plan_iterations
            )

            # Calculate compression effectiveness
            original_count = len(messages)
            optimized_count = len(optimized_messages)
            compression_ratio = (
                (original_count - optimized_count) / original_count * 100
            )

            logger.info(
                f"Planning context optimization completed: "
                f"{original_count} -> {optimized_count} messages "
                f"({compression_ratio:.1f}% reduction)"
            )

            update_dict.update(
                {
                    "messages": optimized_messages,
                    "context_optimization_applied": True,
                    "original_message_count": original_count,
                    "optimized_message_count": optimized_count,
                }
            )

        # Optimize observations using advanced management
        if len(observations) > 5:
            try:
                optimized_observations = context_manager.manage_observations_advanced(
                    observations,
                    optimization_level="standard",  # Use standard optimization for planning
                )

                update_dict["observations"] = optimized_observations
                update_dict["optimized_observations_count"] = len(
                    optimized_observations
                )
                update_dict["original_observations_count"] = len(observations)

                logger.info(
                    f"Observations optimization completed: "
                    f"{len(observations)} -> {len(optimized_observations)} observations"
                )
            except Exception as e:
                logger.error(f"Observations optimization failed: {e}")
                # Fallback: keep the last 5
                update_dict["observations"] = observations[-5:]
                update_dict["observations_optimization_fallback"] = True

        # Update state and jump to planner
        # Preserve research_topic and resources from coordinator_node
        update_dict.update(
            {
                "research_topic": state.get("research_topic", ""),
                "resources": state.get("resources", []),
            }
        )
        return Command(update=update_dict, goto="planner")

    except Exception as e:
        logger.error(f"Planning context optimization failed: {e}")
        # Fallback to unified message optimization
        try:
            from src.utils.context.execution_context_manager import (
                ExecutionContextManager,
                ContextConfig,
            )

            fallback_config = ContextConfig(
                max_context_steps=5,
                max_step_content_length=1000,
                max_observations_length=3000,
                enable_content_deduplication=True,
                enable_smart_truncation=True,
            )

            fallback_manager = ExecutionContextManager(fallback_config)
            truncated_messages = fallback_manager.optimize_messages(
                messages, max_messages=10, max_tokens=20000
            )

            return Command(
                update={
                    "messages": truncated_messages,
                    "context_optimization_fallback": True,
                    "fallback_reason": str(e),
                    # Preserve research_topic and resources from coordinator_node
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                },
                goto="planner",
            )
        except Exception as fallback_error:
            logger.error(f"Fallback optimization also failed: {fallback_error}")
            # Final fallback: use original messages directly
            return Command(
                update={
                    # Preserve research_topic and resources from coordinator_node
                    "research_topic": state.get("research_topic", ""),
                    "resources": state.get("resources", []),
                },
                goto="planner",
            )


# Export all node functions for testing and external use
__all__ = [
    "get_configuration_from_config",
    "background_investigation_node",
    "planner_node",
    "human_feedback_node",
    "coordinator_node",
    "reporter_node",
    "researcher_node",
    "context_optimizer_node",
    "planning_context_optimizer_node",
    "_execute_agent_step",
    "_setup_and_execute_agent_step",
    "create_agent",
    "create_agent_with_managed_prompt",
    "handoff_to_planner",
    # New refactored nodes for EnhancedResearcher decomposition
    "prepare_research_step_node",
    "researcher_agent_node",
    "reflection_node",
    "update_plan_node",
    "check_research_completion_node",
]


# ============================================================================
# REFACTORED NODES: EnhancedResearcher Decomposition
# ============================================================================
# The following nodes replace the monolithic EnhancedResearcher.execute() method
# with single-responsibility graph nodes for better testability and maintainability.


async def prepare_research_step_node(state: State, config: RunnableConfig) -> State:
    """Prepare context and tools for the next research step.

    This node replaces the preparation logic from EnhancedResearcher.__init__
    and the early parts of execute() method.

    Args:
        state: Current graph state
        config: Runnable configuration

    Returns:
        Updated state with agent_input prepared for researcher_agent_node
    """
    logger.info("--- ENTERING PREPARE_RESEARCH_STEP_NODE ---")
    logger.debug(f"Initial state keys: {state.keys()}")

    try:
        # 1. Find current step to execute
        current_plan = state.get("current_plan")
        if not current_plan or not hasattr(current_plan, "steps"):
            logger.warning("No current plan found in state.")
            return_value = {
                "agent_input": None,
                "preparation_error": "No current plan found",
            }
            logger.debug(f"Returning from prepare_research_step_node: {return_value}")
            logger.info("--- EXITING PREPARE_RESEARCH_STEP_NODE ---")
            return return_value

        if not current_plan.steps:
            logger.warning("No steps in current plan.")
            return_value = {
                "agent_input": None,
                "preparation_error": "No steps in current plan",
            }
            logger.debug(f"Returning from prepare_research_step_node: {return_value}")
            logger.info("--- EXITING PREPARE_RESEARCH_STEP_NODE ---")
            return return_value

        current_step = None
        for step in current_plan.steps:
            if not step.execution_res:
                current_step = step
                break

        logger.debug(
            f"Found current step to execute: {current_step.title if current_step else 'None'}"
        )

        if not current_step:
            logger.info("All research steps are completed.")
            return_value = {"agent_input": None, "all_steps_completed": True}
            logger.debug(f"Returning from prepare_research_step_node: {return_value}")
            logger.info("--- EXITING PREPARE_RESEARCH_STEP_NODE ---")
            return return_value

        # 2. Prepare research tools (migrated from EnhancedResearcher._prepare_research_tools)
        from src.tools.search import get_web_search_tool
        from src.tools.crawl import crawl_tool
        from src.tools.retriever import get_retriever_tool

        configurable = get_configuration_from_config(config)
        tools = [
            get_web_search_tool(
                configurable.agents.max_search_results,
                configurable.content.enable_smart_filtering,
            ),
            crawl_tool,
        ]

        # Add retriever tool if resources are available
        resources = state.get("resources")
        if resources:
            tools.insert(0, get_retriever_tool(resources))

        logger.debug(f"Prepared {len(tools)} tools for the agent.")

        # 3. Build agent input (migrated from EnhancedResearcher._build_agent_state)
        from langchain_core.messages import HumanMessage

        agent_input = {
            "messages": [HumanMessage(content=current_step.description)],
            "tools": tools,
            "current_step": current_step,
            "current_plan": current_plan,
        }

        logger.debug(f"Constructed agent_input for step: '{current_step.title}'")

        return_value = {
            "agent_input": agent_input,
            "tools": tools,
            "current_step": current_step,
        }
        logger.debug(
            f"Returning from prepare_research_step_node: {return_value.keys()}"
        )
        logger.info("--- EXITING PREPARE_RESEARCH_STEP_NODE ---")
        return return_value

    except Exception as e:
        logger.error(f"Error in prepare_research_step_node: {e}", exc_info=True)
        return_value = {
            "agent_input": None,
            "preparation_error": f"An unexpected error occurred: {e}",
        }
        logger.debug(
            f"Returning from prepare_research_step_node on error: {return_value}"
        )
        logger.info("--- EXITING PREPARE_RESEARCH_STEP_NODE ---")
        return return_value


async def researcher_agent_node(state: State, config: RunnableConfig) -> State:
    """Execute the researcher agent with prepared input.

    This node replaces the agent execution logic from EnhancedResearcher.execute().

    Args:
        state: Current graph state with agent_input
        config: Runnable configuration

    Returns:
        Updated state with research_result
    """
    logger.info("Invoking the researcher agent.")

    try:
        agent_input = state.get("agent_input")
        if not agent_input:
            error_msg = state.get("preparation_error", "Agent input not prepared")
            logger.error(f"Cannot execute agent: {error_msg}")
            return {"research_result": None, "agent_error": error_msg}

        # Create and execute researcher agent
        configurable = get_configuration_from_config(config)
        agent = create_agent(
            agent_name="researcher",
            agent_type="researcher",
            tools=agent_input["tools"],
            prompt_template="researcher",
            configurable=configurable,
        )

        # Execute with recursion limit
        import os

        default_recursion_limit = 10
        try:
            env_value_str = os.getenv(
                "AGENT_RECURSION_LIMIT", str(default_recursion_limit)
            )
            recursion_limit = (
                int(env_value_str)
                if int(env_value_str) > 0
                else default_recursion_limit
            )
        except (ValueError, TypeError):
            recursion_limit = default_recursion_limit

        logger.info(f"Executing agent with recursion limit: {recursion_limit}")

        from src.llms.error_handler import safe_llm_call_async

        # Create clean input without tools (tools are already bound to agent)
        clean_input = {k: v for k, v in agent_input.items() if k != "tools"}

        result = await safe_llm_call_async(
            agent.ainvoke,
            input=clean_input,
            config={"recursion_limit": recursion_limit},
            operation_name="researcher executor",
            context=f"Execute step: {agent_input.get('current_step_title', 'Unknown')}",
            enable_smart_processing=True,
            max_retries=3,
        )

        # Extract response content
        if hasattr(result, "get") and "messages" in result:
            response_content = result["messages"][-1].content
        else:
            response_content = (
                result.content if hasattr(result, "content") else str(result)
            )

        logger.info(
            f"Agent execution completed for step: {agent_input.get('current_step_title')}"
        )
        return {"research_result": response_content}

    except Exception as e:
        logger.error(f"Error in researcher agent execution: {e}", exc_info=True)
        return {"research_result": None, "agent_error": str(e)}


async def reflection_node(state: State, config: RunnableConfig) -> dict:
    """
    Analyzes the research results for knowledge gaps and determines if the research is sufficient.
    This node uses an EnhancedReflectionAgent to perform the analysis.
    """
    logger.info("--- ENTERING REFLECTION_NODE ---")

    # 1. Get configuration and initialize reflection agent
    configurable = get_configuration_from_config(config)
    from src.utils.reflection.enhanced_reflection import EnhancedReflectionAgent
    from src.utils.reflection.reflection_manager import get_reflection_manager

    # Get session_id for session-specific reflection tracking
    session_id = (
        config.get("configurable", {}).get("thread_id")
        if isinstance(config, dict) and "configurable" in config
        else state.get("thread_id", "default")
    )

    # Get the reflection manager instance, configured with max_loops
    max_reflection_loops = getattr(configurable.reflection, "max_loops", 2)
    reflection_manager = get_reflection_manager(max_reflections=max_reflection_loops)

    # Log the current reflection count check
    logger.info(
        f"Reflection check - Session: {session_id}, "
        f"Current count: {reflection_manager.get_session_stats(session_id).total_reflections}, "
        f"Max allowed: {max_reflection_loops}"
    )

    # Pre-execution check to see if reflection is allowed
    can_execute, reason = reflection_manager.can_execute_reflection(
        session_id=session_id, reflection_type="iteration"
    )

    if not can_execute:
        logger.info(f"Reflection execution blocked: {reason}")
        return {
            "reflection_insights": {
                "is_sufficient": True,
                "reason": reason,
                "primary_follow_up_query": None,
            }
        }

    reflection_agent = EnhancedReflectionAgent(config=configurable)

    # 2. Prepare context for reflection
    from src.utils.reflection.enhanced_reflection import ReflectionContext

    research_result = state.get("research_result", "")
    research_topic = state.get("research_topic", "")
    observations = state.get("observations", [])
    current_step = state.get("current_step", None)
    current_plan = state.get("current_plan")

    context = ReflectionContext(
        research_topic=research_topic,
        completed_steps=[
            step.model_dump()
            for step in current_plan.steps
            if step.execution_res is not None
        ],
        current_step=current_step.model_dump() if current_step else None,
        execution_results=[research_result],
        observations=observations,
        locale=state.get("locale", "en-US"),
        total_steps=len(current_plan.steps) if current_plan else 0,
        current_step_index=(
            current_plan.steps.index(current_step)
            if current_step and current_plan
            else 0
        ),
    )

    # 3. Analyze for knowledge gaps
    logger.info("Analyzing knowledge gaps with EnhancedReflectionAgent.")
    reflection_insights = await reflection_agent.analyze_knowledge_gaps(context)

    # 4. Record the reflection execution
    reflection_manager.record_reflection(
        session_id=session_id, reflection_type="iteration"
    )
    logger.info(
        f"Reflection recorded. Session: {session_id}, "
        f"Count: {reflection_manager.get_session_stats(session_id).total_reflections}"
    )

    # 5. Log the results
    if reflection_insights:
        logger.info(
            f"Reflection complete. Sufficient: {reflection_insights.is_sufficient}"
        )
        logger.debug(
            f"Reflection insights: {reflection_insights.model_dump_json(indent=2)}"
        )
    else:
        logger.warning("Reflection analysis did not return any insights.")

    logger.info("--- EXITING REFLECTION_NODE ---")
    return {"reflection_insights": reflection_insights}


async def update_plan_node(state: State, config: RunnableConfig) -> dict:
    """
    Updates the research plan based on the reflection insights.
    If new queries are suggested, they are added as new steps to the plan.
    The reflection count is now managed by ReflectionManager and is no longer handled here.
    """
    logger.info("--- ENTERING UPDATE_PLAN_NODE ---")

    current_plan = state.get("current_plan")
    reflection_insights = state.get("reflection_insights")

    if (
        reflection_insights
        and hasattr(reflection_insights, "primary_follow_up_query")
        and reflection_insights.primary_follow_up_query
    ):
        logger.info(
            f"Found follow-up query to add to the plan: {reflection_insights.primary_follow_up_query}"
        )

        new_step = Step(
            need_search=True,
            step_type=StepType.RESEARCH,
            title=f"Follow-up Research: {reflection_insights.primary_follow_up_query[:100]}",
            description=f"Investigate the following query based on reflection analysis: {reflection_insights.primary_follow_up_query}",
            execution_res=None,
            )

        if current_plan and hasattr(current_plan, "steps"):
            updated_steps = current_plan.steps + [new_step]
            updated_plan = current_plan.copy(update={"steps": updated_steps})
            logger.info(f"Added 1 new step to the plan.")
        else:
            updated_plan = Plan(
                title=state.get("research_topic", "Follow-up Research Plan"),
                thought="This plan was generated to address knowledge gaps identified during reflection.",
                steps=new_steps,
            )
            logger.info("Created a new plan with follow-up steps.")

        # The reflection state is preserved but the count is no longer managed here.
        updated_reflection_state = state.get("reflection", {})

        logger.info("--- EXITING UPDATE_PLAN_NODE (with plan update) ---")
        return {
            "current_plan": updated_plan,
            "reflection": updated_reflection_state,
        }

    logger.info("No follow-up queries needed. Plan remains unchanged.")
    logger.info("--- EXITING UPDATE_PLAN_NODE (no update) ---")
    return {}


def check_research_completion_node(state: State) -> Literal["research_team", "planner"]:
    """Check if all research steps are completed and decide next action.

    This node replaces the completion check logic from the original workflow.

    Args:
        state: Current graph state

    Returns:
        Next node to execute: "research_team" for more research, "planner" if done
    """
    logger.info("Checking research completion status.")

    try:
        # Check for errors first
        if (
            state.get("preparation_error")
            or state.get("agent_error")
            or state.get("update_error")
        ):
            logger.warning(
                "Error detected, continuing to research_team for error handling"
            )
            return "research_team"

        # Check if all steps are completed
        if state.get("all_steps_completed"):
            logger.info("All research steps completed, proceeding to planner")
            return "planner"

        # Check if more research is needed based on reflection
        if state.get("needs_more_research"):
            logger.info(
                "Reflection indicates more research needed, continuing research loop"
            )
            return "research_team"

        # Check current plan status
        current_plan = state.get("current_plan")
        if current_plan and hasattr(current_plan, "steps"):
            remaining_steps = [
                step for step in current_plan.steps if not step.execution_res
            ]
            if remaining_steps:
                logger.info(
                    f"{len(remaining_steps)} steps remaining, continuing research"
                )
                return "research_team"
            else:
                logger.info("All plan steps completed, proceeding to planner")
                return "planner"
        else:
            logger.warning("No current plan found, proceeding to planner")
            return "planner"

    except Exception as e:
        logger.error(f"Error checking research completion: {e}", exc_info=True)
        return "research_team"  # Safe fallback
