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
    crawl_tool,
    python_repl_tool,
)
from src.tools.search import get_web_search_tool
from src.tools.retriever import get_retriever_tool

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

                # Set agents configuration with overrides
                if "max_search_results" in self._overrides:
                    self.agents.max_search_results = self._overrides[
                        "max_search_results"
                    ]
                else:
                    self.agents.max_search_results = getattr(
                        base_settings.agents, "max_search_results", 5
                    )

                if "enable_deep_thinking" in self._overrides:
                    self.agents.enable_deep_thinking = self._overrides[
                        "enable_deep_thinking"
                    ]
                else:
                    self.agents.enable_deep_thinking = getattr(
                        base_settings.agents, "enable_deep_thinking", False
                    )

                if "max_plan_iterations" in self._overrides:
                    self.agents.max_plan_iterations = self._overrides[
                        "max_plan_iterations"
                    ]
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

                # Enhanced reflection configuration with overrides
                if "enable_enhanced_reflection" in self._overrides:
                    self.enable_enhanced_reflection = self._overrides[
                        "enable_enhanced_reflection"
                    ]
                else:
                    self.enable_enhanced_reflection = getattr(
                        base_settings, "enable_enhanced_reflection", True
                    )
                self.max_reflection_loops = getattr(
                    base_settings, "max_reflection_loops", 1
                )

                # Set reflection configuration with override support
                if "reflection_model" in self._overrides:
                    self.reflection.reflection_model = self._overrides[
                        "reflection_model"
                    ]
                else:
                    reflection_model = getattr(
                        base_settings.reflection, "reflection_model", None
                    )
                    if reflection_model is None:
                        # Use reasoning model if enabled, otherwise use basic model
                        if self.agents.enable_deep_thinking:
                            self.reflection.reflection_model = "reasoning"
                        else:
                            self.reflection.reflection_model = "basic"
                    else:
                        self.reflection.reflection_model = reflection_model

                # Also set it at the top level for backward compatibility
                self.reflection_model = self.reflection.reflection_model

                self.knowledge_gap_threshold = getattr(
                    base_settings, "knowledge_gap_threshold", 0.7
                )
                self.sufficiency_threshold = getattr(
                    base_settings, "sufficiency_threshold", 0.8
                )
                self.enable_reflection_integration = getattr(
                    base_settings, "enable_reflection_integration", True
                )

                # Reflection integration configuration for initial stage skipping
                self.reflection.skip_initial_stage_reflection = getattr(
                    base_settings.reflection, "skip_initial_stage_reflection", True
                )
                # Also set them at the top level for backward compatibility
                self.skip_initial_stage_reflection = (
                    self.reflection.skip_initial_stage_reflection
                )

                # Max step num configuration with override - prioritize API override
                if "max_step_num" in self._overrides:
                    self.max_step_num = self._overrides["max_step_num"]
                else:
                    self.max_step_num = getattr(base_settings.agents, "max_step_num", 3)

                # Resources configuration with override
                if "resources" in self._overrides:
                    self.resources = self._overrides["resources"]
                else:
                    self.resources = getattr(base_settings, "resources", [])

                # MCP configuration
                self.mcp = getattr(
                    base_settings,
                    "mcp",
                    type("mcp", (), {"enabled": False, "servers": [], "timeout": 30})(),
                )

        if "configurable" in config:
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
        "enable_enhanced_reflection": unified_config.enable_enhanced_reflection,
        "max_reflection_loops": unified_config.max_reflection_loops,
        "reflection_model": unified_config.reflection_model,
        "knowledge_gap_threshold": unified_config.knowledge_gap_threshold,
        "sufficiency_threshold": unified_config.sufficiency_threshold,
        "enable_reflection_integration": unified_config.enable_reflection_integration,
    }

    # Phase 5: Initialize reflection agent for planner if enabled
    reflection_agent = None
    if reflection_config.get("enable_enhanced_reflection", True):
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
    if reflection_insights and reflection_config.get(
        "enable_reflection_integration", True
    ):
        logger.info("Integrating reflection insights into planning")

        # Add reflection context to planning messages
        reflection_context = f"""
        
REFLECTION INSIGHTS FROM PREVIOUS RESEARCH:
        - Knowledge Gaps Identified: {len(reflection_insights.get('knowledge_gaps', []))}
        - Research Sufficiency: {'Sufficient' if reflection_insights.get('is_sufficient', False) else 'Insufficient'}
        - Suggested Follow-up Queries: {reflection_insights.get('follow_up_queries', [])}
        - Confidence Score: {reflection_insights.get('confidence_score', 'N/A')}
        
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
                if planner_reflection and planner_reflection.knowledge_gaps:
                    reflection_context += f"""

## Planner Reflection Analysis

**Planning Completeness**: {"Complete" if planner_reflection.is_sufficient else "Incomplete"}
**Planning Confidence**: {planner_reflection.confidence:.2f}

**Planning Gaps Identified**:
{
                        chr(10).join([
                            f"- {gap if isinstance(gap, str) else str(gap.get('query', gap.get('question', gap.get('description', str(gap)))) if isinstance(gap, dict) else str(gap))}"
                            for gap in planner_reflection.follow_up_queries
                        ]) if planner_reflection.follow_up_queries else "- No planning gaps identified"
                    }
                    """.strip()

                    logger.info(
                        f"Planner reflection completed - Sufficient: {planner_reflection.is_sufficient}, Gaps: {len(planner_reflection.knowledge_gaps)}"
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
        return Command(goto="reporter")

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
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")
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
                return Command(goto="reporter")
            else:
                return Command(goto="__end__")
        # Phase 5: Add reflection metadata to command result
        reflection_metadata = {
            "reflection_applied": (
                reflection_config.get("enable_reflection_integration", True)
                and reflection_insights is not None
            ),
            "timestamp": datetime.now().isoformat(),
            "planner_reflection_enabled": reflection_agent is not None,
            "reflection_config": reflection_config,
        }

        if reflection_insights:
            reflection_metadata.update(
                {
                    "knowledge_gaps_count": len(
                        reflection_insights.get("knowledge_gaps", [])
                    ),
                    "confidence": reflection_insights.get("confidence_score", 0.0),
                    "is_sufficient": reflection_insights.get("is_sufficient", False),
                }
            )

        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
                "reflection_metadata": reflection_metadata,
                "plan_iterations": plan_iterations + 1,
            },
            goto="reporter",
        )

    # Phase 5: Add reflection metadata for human feedback path as well
    reflection_metadata = {
        "reflection_applied": (
            reflection_config.get("enable_reflection_integration", True)
            and reflection_insights is not None
        ),
        "timestamp": datetime.now().isoformat(),
        "planner_reflection_enabled": reflection_agent is not None,
        "reflection_config": reflection_config,
    }

    if reflection_insights:
        reflection_metadata.update(
            {
                "knowledge_gaps_count": len(
                    reflection_insights.get("knowledge_gaps", [])
                ),
                "confidence": reflection_insights.get("confidence_score", 0.0),
                "is_sufficient": reflection_insights.get("is_sufficient", False),
            }
        )

    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
            "reflection_metadata": reflection_metadata,
            "plan_iterations": plan_iterations + 1,
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
            goto = "reporter"
    except json.JSONDecodeError as e:
        logger.warning(f"Planner response is not a valid JSON: {e}")
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        if plan_iterations > 1:  # the plan_iterations is increased before this check
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    try:
        validated_plan = Plan.model_validate(new_plan)
    except Exception as e:
        logger.warning(
            f"Human feedback node execution error: Failed to parse Plan from completion {new_plan}. Got: {e}"
        )
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        if plan_iterations > 1:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    return Command(
        update={
            "current_plan": validated_plan,
            "plan_iterations": plan_iterations,
            "locale": new_plan["locale"],
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
        logger.info("All steps completed, moving to planner")
        return Command(goto="planner")

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
    """Execute steps in parallel using the ParallelExecutor with rate limiting."""
    logger.info(
        f"Starting parallel execution with max {max_parallel_tasks} concurrent tasks"
    )

    from src.utils.performance.parallel_executor import (
        create_parallel_executor,
        ParallelTask,
        TaskPriority,
    )

    current_plan = state.get("current_plan")
    steps = current_plan.steps

    # Analyze dependencies
    dependencies = _analyze_step_dependencies(steps)
    logger.info(f"Step dependencies: {dependencies}")

    observations = list(state.get("observations", []))
    all_messages = []

    # Create parallel executor with rate limiting and shared context
    from src.utils.performance.parallel_executor import SharedTaskContext

    shared_context = SharedTaskContext()
    executor = create_parallel_executor(
        max_concurrent_tasks=max_parallel_tasks,
        enable_rate_limiting=True,
        enable_adaptive_scheduling=True,
        shared_context=shared_context,
    )

    # Convert steps to parallel tasks
    parallel_tasks = []
    for step in steps:
        if step.execution_res:  # Skip already completed steps
            continue

        # Determine agent type and priority
        if step.step_type == StepType.RESEARCH:
            agent_type = "researcher"
            priority = TaskPriority.NORMAL
        elif step.step_type == StepType.PROCESSING:
            agent_type = "coder"
            priority = TaskPriority.HIGH  # Processing steps often depend on research
        else:
            agent_type = "researcher"
            priority = TaskPriority.NORMAL

        # Map dependencies to task IDs
        task_dependencies = []
        for dep_title in dependencies.get(step.title, set()):
            task_dependencies.append(dep_title)

        task = ParallelTask(
            task_id=step.title,
            func=_execute_single_step_parallel,
            args=(state, config, step, agent_type),
            priority=priority,
            max_retries=2,  # Allow retries for failed steps
            timeout=500.0,  # 5 minutes timeout per step
            dependencies=task_dependencies,
        )
        parallel_tasks.append(task)

    if not parallel_tasks:
        logger.info("No tasks to execute, all steps completed")
        return Command(goto="planner")

    # Add tasks to executor
    executor.add_tasks(parallel_tasks)

    # Execute all tasks
    try:
        results = await executor.execute_all()

        # Process results and create updated steps list
        updated_steps = list(steps)  # Create a copy of the steps list
        for task_id, task_result in results.items():
            step_index = next(
                (i for i, s in enumerate(steps) if s.title == task_id), None
            )
            if step_index is None:
                continue

            step = steps[step_index]
            if task_result.status.value == "completed":
                logger.info(f"Step '{task_id}' completed successfully")
                if task_result.result and hasattr(task_result.result, "content"):
                    observations.append(task_result.result.content)
                    # Get the agent type from the step
                    agent_type = (
                        "researcher" if step.step_type == StepType.RESEARCH else "coder"
                    )
                    all_messages.append(
                        AIMessage(content=task_result.result.content, name=agent_type)
                    )

                    # Update the step with the new one containing execution result
                    if hasattr(task_result.result, "updated_step"):
                        updated_steps[step_index] = task_result.result.updated_step
            else:
                logger.error(
                    f"Step '{task_id}' failed with status: {task_result.status.value}"
                )
                if task_result.error:
                    logger.error(f"Error details: {task_result.error}")
                # Ensure step has some result to prevent infinite loop
                if not step.execution_res:
                    error_msg = (
                        str(task_result.error)
                        if task_result.error
                        else "Execution failed"
                    )
                    # Create updated step with error message (immutable pattern)
                    updated_steps[step_index] = step.copy(
                        update={"execution_res": f"Execution failed: {error_msg}"}
                    )

        # Log execution statistics
        stats = executor.get_stats()
        logger.info(f"Parallel execution stats: {stats}")

    except Exception as e:
        logger.error(f"Parallel executor failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Mark all unexecuted steps as failed (immutable pattern)
        if "updated_steps" not in locals():
            updated_steps = list(steps)  # Create copy if not already created
        for i, step in enumerate(steps):
            if not step.execution_res:
                updated_steps[i] = step.copy(
                    update={"execution_res": f"Parallel Executor failed: {str(e)}"}
                )

    # Create updated plan with new steps
    updated_plan = current_plan.copy(update={"steps": updated_steps})

    # Check if all steps are completed
    if all(step.execution_res for step in updated_steps):
        logger.info("All steps completed via parallel execution")
        return Command(
            update={
                "messages": all_messages,
                "observations": observations,
                "current_plan": updated_plan,
            },
            goto="planner",
        )
    else:
        # Some steps still pending, continue
        return Command(
            update={
                "messages": all_messages,
                "observations": observations,
                "current_plan": updated_plan,
            },
            goto="research_team",
        )


async def _execute_single_step_parallel(
    state: State, config: RunnableConfig, step: Step, agent_type: str
) -> Any:
    """Execute a single step in parallel context through researcher_node or coder_node.

    This function now routes through the appropriate node functions to ensure
    consistent behavior between parallel and serial execution, including
    context isolation, reflection integration, and other advanced features.
    """
    logger.info(f"Executing step in parallel: {step.title} with {agent_type}")

    # Create a temporary state that contains only the current step to execute
    current_plan = state.get("current_plan")

    # Create a new plan with only the current step as the first unexecuted step
    temp_steps = []
    for s in current_plan.steps:
        if s.title == step.title:
            # This is the step we want to execute - make sure it's not marked as executed
            temp_steps.append(s.copy(update={"execution_res": None}))
        elif s.execution_res:
            # Keep completed steps as they are for context
            temp_steps.append(s)
        else:
            # Mark other unexecuted steps as completed to avoid execution
            temp_steps.append(
                s.copy(update={"execution_res": "Pending parallel execution"})
            )

    temp_plan = current_plan.copy(update={"steps": temp_steps})
    temp_state = state.copy()
    temp_state["current_plan"] = temp_plan

    # Route to appropriate node based on agent type
    try:
        if agent_type == "researcher":
            result = await researcher_node(temp_state, config)
        elif agent_type == "coder":
            result = await coder_node(temp_state, config)
        else:
            # Default to researcher for unknown types
            result = await researcher_node(temp_state, config)

        # Extract the execution result from the node's response
        if hasattr(result, "update") and result.update:
            updated_plan = result.update.get("current_plan")
            if updated_plan:
                # Find the executed step in the updated plan
                for updated_step in updated_plan.steps:
                    if updated_step.title == step.title and updated_step.execution_res:
                        # Create the result object expected by parallel executor
                        response_content = updated_step.execution_res
                        final_step = step.copy(
                            update={"execution_res": response_content}
                        )

                        logger.info(
                            f"Parallel step '{step.title}' completed by {agent_type} through node"
                        )

                        return type(
                            "StepResult",
                            (),
                            {"content": response_content, "updated_step": final_step},
                        )()

        # Fallback: if we can't extract the result properly, mark as completed with basic info
        logger.warning(
            f"Could not extract proper result for step '{step.title}', using fallback"
        )
        fallback_content = f"Step completed by {agent_type} (result extraction failed)"
        fallback_step = step.copy(update={"execution_res": fallback_content})

        return type(
            "StepResult",
            (),
            {"content": fallback_content, "updated_step": fallback_step},
        )()

    except Exception as e:
        logger.error(
            f"Error executing step '{step.title}' through {agent_type} node: {e}"
        )
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Create error result
        error_content = f"Execution failed: {str(e)}"
        error_step = step.copy(update={"execution_res": error_content})

        return type(
            "StepResult", (), {"content": error_content, "updated_step": error_step}
        )()


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
        "messages": [],  # Start with empty messages for prompt system
    }

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


async def researcher_node_with_isolation(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node with context isolation - Phase 3 Simplified Implementation.

    This function implements the enhanced researcher node with context isolation
    using the EnhancedResearcher class for better maintainability and modularity.
    """
    # Phase 3 Simplification: Import EnhancedResearcher
    from src.utils.researcher import EnhancedResearcher

    logger.info("Executing researcher node with context isolation (Phase 3 Optimized)")

    try:
        # Phase 3: Create and initialize enhanced researcher
        researcher = EnhancedResearcher(state, config)

        # Execute complete research workflow with reflection
        result = await researcher.execute_research_with_reflection()

        logger.info("Enhanced researcher node completed successfully")
        return result

    except Exception as e:
        logger.error(f"Enhanced researcher node failed: {e}")
        logger.warning("Falling back to standard researcher node")

        # Fallback to standard researcher node
        return await researcher_node(state, config)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research with context isolation capabilities.

    Phase 1 Implementation: Enhanced with ResearcherContextExtension for
    context isolation to prevent context accumulation in parallel execution.
    """
    logger.info("Researcher node is researching.")
    configurable = get_configuration_from_config(config)

    # Check if context isolation is enabled
    enable_researcher_isolation = getattr(
        configurable, "enable_researcher_isolation", True
    )

    if enable_researcher_isolation:
        logger.info("Using researcher node with context isolation")
        return await researcher_node_with_isolation(state, config)
    else:
        print("DEBUG: === Using standard researcher node ===")
        logger.info("Using standard researcher node")
        tools = [
            get_web_search_tool(
                configurable.agents.max_search_results,
                configurable.content.enable_smart_filtering,
            ),
            crawl_tool,
        ]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)
        logger.info(f"Researcher tools: {tools}")
        return await _setup_and_execute_agent_step(
            state,
            config,
            "researcher",
            tools,
        )


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

        return Command(update={"observations": optimized_observations}, goto="reporter")

    except Exception as e:
        logger.error(f"Context optimization failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.warning(
            "Proceeding with original observations due to optimization failure."
        )

        # Fallback: proceed with original observations if optimization fails
        return Command(goto="reporter")


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
                },
                goto="planner",
            )
        except Exception as fallback_error:
            logger.error(f"Fallback optimization also failed: {fallback_error}")
            # Final fallback: use original messages directly
            return Command(goto="planner")


# Export all node functions for testing and external use
__all__ = [
    "get_configuration_from_config",
    "background_investigation_node",
    "planner_node",
    "human_feedback_node",
    "coordinator_node",
    "reporter_node",
    "researcher_node",
    "researcher_node_with_isolation",
    "context_optimizer_node",
    "planning_context_optimizer_node",
    "_execute_agent_step",
    "_setup_and_execute_agent_step",
    "create_agent",
    "create_agent_with_managed_prompt",
    "handoff_to_planner",
]
