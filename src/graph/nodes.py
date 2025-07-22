# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Annotated, Literal, List, Dict, Any, Set

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

# Delayed import to avoid circular import
# from src.agents import create_agent, create_agent_with_managed_prompt
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.config_loader import get_settings
from src.config.models import SearchEngine

# ExecutionContextManager will be imported dynamically to avoid circular imports
from src.llms.error_handler import safe_llm_call, safe_llm_call_async
from src.llms.llm import get_llm_by_type
from src.utils.template import apply_prompt_template
from src.utils.common.json_utils import repair_json_output

from .types import State
from src.models.planner_model import Plan, Step, StepType
from src.utils.reflection.enhanced_reflection import ReflectionResult
from src.utils.reflection.reflection_integration import (
    ReflectionIntegrationConfig,
)

logger = logging.getLogger(__name__)


def get_configuration_from_config(config):
    """Helper function to get configuration from settings with config overrides."""
    from src.config import get_settings

    try:
        # Get base settings
        base_settings = get_settings()

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
                    base_settings, "max_reflection_loops", 3
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

    # Debug: Log the max_step_num value from configurable
    max_step_num_value = getattr(configurable, "max_step_num", "NOT_FOUND")
    logger.info(f"DEBUG: configurable.max_step_num = {max_step_num_value}")

    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    messages = apply_prompt_template("planner", state, configurable)

    # Phase 5: Enhanced reflection configuration for planner
    reflection_config = {
        "enable_enhanced_reflection": getattr(
            configurable, "enable_enhanced_reflection", True
        ),
        "max_reflection_loops": getattr(configurable, "max_reflection_loops", 3),
        "reflection_model": getattr(configurable.reflection, "reflection_model", None)
        or (
            "reasoning"
            if getattr(configurable.agents, "enable_deep_thinking", False)
            else "basic"
        ),
        "knowledge_gap_threshold": getattr(
            configurable, "knowledge_gap_threshold", 0.7
        ),
        "sufficiency_threshold": getattr(configurable, "sufficiency_threshold", 0.8),
        "enable_reflection_integration": getattr(
            configurable, "enable_reflection_integration", True
        ),
    }

    # Phase 5: Initialize reflection agent for planner if enabled
    reflection_agent = None
    if reflection_config.get("enable_enhanced_reflection", True):
        try:
            from src.utils.reflection.enhanced_reflection import (
                EnhancedReflectionAgent,
                ReflectionConfig,
            )

            ReflectionConfig(
                enable_enhanced_reflection=True,
                max_reflection_loops=reflection_config["max_reflection_loops"],
                reflection_model=reflection_config["reflection_model"],
                knowledge_gap_threshold=reflection_config["knowledge_gap_threshold"],
                sufficiency_threshold=reflection_config["sufficiency_threshold"],
            )

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
{chr(10).join([f"- {gap}" for gap in planner_reflection.follow_up_queries]) if planner_reflection.follow_up_queries else "- No planning gaps identified"}
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

    # Get parallel execution configuration
    configurable = get_configuration_from_config(config)
    enable_parallel = getattr(configurable, "enable_parallel_execution", True)
    max_parallel_tasks = getattr(configurable, "max_parallel_tasks", 3)

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
            timeout=300.0,  # 5 minutes timeout per step
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

    # Initialize unified context manager with configuration
    configurable = get_configuration_from_config(config)
    context_config = ContextConfig(
        max_context_steps=getattr(configurable, "max_context_steps_parallel", 3),
        max_step_content_length=2000,
        max_observations_length=10000,
        enable_content_deduplication=True,
        enable_smart_truncation=True,
    )

    # Check if context sharing should be disabled
    disable_context_parallel = getattr(configurable, "disable_context_parallel", False)
    if disable_context_parallel:
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
        agent_state["citation_reminder"] = "IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)"

    # Invoke the agent
    default_recursion_limit = 25
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
        observations + [new_observation],
        max_observations=context_config.max_observations_length,
        compression_threshold=context_config.max_step_content_length,
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
    if mcp_servers:
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

            agent = create_agent(agent_type, agent_type, loaded_tools, agent_type)
            return await _execute_agent_step(state, config, agent, agent_type)
    else:
        # Use default tools if no MCP servers are configured
        # Delayed import to avoid circular import
        from src.agents import create_agent

        agent = create_agent(agent_type, agent_type, default_tools, agent_type)
        return await _execute_agent_step(state, config, agent, agent_type)


async def researcher_node_with_isolation(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node with context isolation - Phase 4 Implementation.

    This function implements the enhanced researcher node with context isolation
    capabilities including progressive enablement, monitoring, and reflection integration.
    """
    from src.utils.researcher.researcher_context_extension import (
        ResearcherContextExtension,
        ResearcherContextConfig,
    )

    logger.info("Executing researcher node with context isolation (Phase 4)")
    configurable = get_configuration_from_config(config)

    # Configure isolation level based on configuration
    isolation_level = getattr(configurable, "researcher_isolation_level", "moderate")
    isolation_config = ResearcherContextConfig(
        max_context_steps=getattr(configurable, "max_context_steps_researcher", 2),
        max_step_content_length=1500,
        max_observations_length=8000,
        isolation_level=isolation_level,
    )

    # Phase 3: Prepare configuration for progressive enablement and metrics
    phase3_config = {
        "researcher_isolation_metrics": getattr(
            configurable, "researcher_isolation_metrics", False
        ),
        "researcher_auto_isolation": getattr(
            configurable, "researcher_auto_isolation", False
        ),
        "researcher_isolation_threshold": getattr(
            configurable, "researcher_isolation_threshold", 0.7
        ),
        "researcher_max_local_context": getattr(
            configurable, "researcher_max_local_context", 3000
        ),
    }

    # Phase 4: Enhanced reflection configuration
    reflection_config = {
        "enable_enhanced_reflection": getattr(
            configurable, "enable_enhanced_reflection", True
        ),
        "max_reflection_loops": getattr(configurable, "max_reflection_loops", 3),
        "reflection_model": getattr(configurable.reflection, "reflection_model", None)
        or (
            "reasoning"
            if getattr(configurable.agents, "enable_deep_thinking", False)
            else "basic"
        ),
        "knowledge_gap_threshold": getattr(
            configurable, "knowledge_gap_threshold", 0.7
        ),
        "sufficiency_threshold": getattr(configurable, "sufficiency_threshold", 0.8),
        "enable_reflection_integration": getattr(
            configurable, "enable_reflection_integration", True
        ),
    }

    # Initialize context extension with Phase 3 features
    context_extension = ResearcherContextExtension(
        isolation_config=isolation_config, config=phase3_config
    )

    # Phase 5: Initialize reflection agent if enabled
    reflection_agent = None
    if reflection_config.get("enable_enhanced_reflection", False):
        logger.info(
            f"Enhanced reflection enabled for researcher_node: {reflection_config}"
        )
        try:
            from src.utils.reflection.enhanced_reflection import (
                EnhancedReflectionAgent,
                ReflectionConfig,
            )

            reflection_agent_config = ReflectionConfig(
                enable_enhanced_reflection=True,
                max_reflection_loops=reflection_config["max_reflection_loops"],
                reflection_model=reflection_config["reflection_model"],
                knowledge_gap_threshold=reflection_config["knowledge_gap_threshold"],
                sufficiency_threshold=reflection_config["sufficiency_threshold"],
            )

            reflection_agent = EnhancedReflectionAgent(config=configurable)

            logger.info("Enhanced reflection agent initialized")

        except Exception as e:
            logger.error(f"Failed to initialize reflection agent: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            reflection_agent = None

    # Setup tools
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

    # Setup MCP servers if configured
    mcp_servers = {}
    enabled_tools = {}

    if configurable.mcp.enabled and configurable.mcp.servers:
        for server_name, server_config in enumerate(configurable.mcp.servers):
            if (
                server_config["enabled_tools"]
                and "researcher" in server_config["add_to_agents"]
            ):
                server_key = f"server_{server_name}"
                mcp_servers[server_key] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_key

    # Create and execute agent with isolation
    if mcp_servers:
        async with MultiServerMCPClient(mcp_servers) as client:
            loaded_tools = tools[:]
            for tool in client.get_tools():
                if tool.name in enabled_tools:
                    tool.description = (
                        f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                    )
                    loaded_tools.append(tool)
            # Delayed import to avoid circular import
            from src.agents import create_agent_with_managed_prompt

            agent = create_agent_with_managed_prompt(
                "researcher", "researcher", loaded_tools
            )
            isolation_result = (
                await context_extension.execute_researcher_step_with_isolation(
                    state, config, agent, "researcher"
                )
            )
    else:
        # Delayed import to avoid circular import
        from src.agents import create_agent_with_managed_prompt

        agent = create_agent_with_managed_prompt("researcher", "researcher", tools)
        isolation_result = (
            await context_extension.execute_researcher_step_with_isolation(
                state, config, agent, "researcher"
            )
        )

    # Phase 5: Enhanced reflection integration with iterative research workflow
    if reflection_agent is not None:
        try:
            logger.info("Applying enhanced reflection with iterative research loop")

            # Iterative research configuration
            max_iterations = getattr(configurable, "max_follow_up_iterations", 3)
            sufficiency_threshold = getattr(configurable, "sufficiency_threshold", 0.7)
            enable_iterative_research = getattr(
                configurable, "enable_iterative_research", True
            )
            max_queries_per_iteration = getattr(
                configurable, "max_queries_per_iteration", 3
            )
            follow_up_delay = getattr(configurable, "follow_up_delay_seconds", 1.0)
            current_iteration = 0

            # Check if iterative research is enabled
            if not enable_iterative_research:
                logger.info(
                    "Iterative research is disabled, performing single reflection analysis"
                )
                max_iterations = 1

            # Initialize research findings and observations
            all_research_findings = []
            all_observations = list(state.get("observations", []))

            # Extract initial research findings from isolation result
            if hasattr(isolation_result, "update") and isolation_result.update:
                observations = isolation_result.update.get("observations", [])
                messages = isolation_result.update.get("messages", [])

                # Add observations to all_observations
                all_observations.extend(observations)

                # Combine findings from both sources
                for obs in observations:
                    if isinstance(obs, dict) and "content" in obs:
                        all_research_findings.append(obs["content"])
                    elif isinstance(obs, str):
                        all_research_findings.append(obs)
                    else:
                        all_research_findings.append(str(obs))

                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        all_research_findings.append(msg["content"])
                    elif hasattr(msg, "content"):
                        all_research_findings.append(msg.content)
                    else:
                        all_research_findings.append(str(msg))

            # Update state with accumulated observations for reflection check
            state["observations"] = all_observations

            # Extract completed steps from current plan
            current_plan = state.get("current_plan")
            completed_steps = []
            if current_plan and hasattr(current_plan, "steps"):
                for step in current_plan.steps:
                    if step.execution_res:
                        completed_steps.append(
                            {
                                "step": step.title,
                                "description": step.description,
                                "execution_res": step.execution_res,
                            }
                        )

            # Start iterative research loop
            logger.info(
                f"Starting iterative research loop (max_iterations: {max_iterations}, threshold: {sufficiency_threshold})"
            )

            while current_iteration < max_iterations:
                current_iteration += 1
                logger.info(
                    f"Iteration {current_iteration}/{max_iterations}: Performing reflection analysis..."
                )

                # Analyze knowledge gaps in current research
                from src.utils.reflection.enhanced_reflection import ReflectionContext

                # Create proper ReflectionContext object with current findings
                reflection_context = ReflectionContext(
                    research_topic=state.get("research_topic", ""),
                    completed_steps=completed_steps,
                    execution_results=all_research_findings,
                    observations=all_observations,
                    total_steps=(
                        len(current_plan.steps)
                        if current_plan and hasattr(current_plan, "steps")
                        else 1
                    ),
                    current_step_index=len(completed_steps),
                    locale=state.get("locale", "en-US"),
                )

                # Initialize ReflectionIntegrator if not already done
                if not hasattr(state, "_reflection_integrator"):
                    from src.utils.reflection.reflection_integration import (
                        ReflectionIntegrator,
                    )

                    reflection_config = ReflectionIntegrationConfig(
                        enable_reflection_integration=config.get(
                            "enable_reflection_integration", True
                        ),
                        reflection_trigger_threshold=config.get(
                            "reflection_trigger_threshold", 3
                        ),
                        skip_initial_stage_reflection=config.get(
                            "skip_initial_stage_reflection", True
                        ),
                        enable_progressive_reflection=config.get(
                            "enable_progressive_reflection", True
                        ),
                        enable_reflection_metrics=config.get(
                            "enable_reflection_metrics", True
                        ),
                    )
                    state["_reflection_integrator"] = ReflectionIntegrator(
                        reflection_agent=reflection_agent, config=reflection_config
                    )

                reflection_integrator = state["_reflection_integrator"]

                # Check if reflection should be triggered
                should_trigger, trigger_reason, decision_factors = (
                    reflection_integrator.should_trigger_reflection(
                        state=state, current_step=None, agent_name="researcher"
                    )
                )

                logger.info(
                    f"Reflection trigger decision: {should_trigger}, reason: {trigger_reason}"
                )

                # Perform reflection analysis only if triggered
                if should_trigger:
                    reflection_result = await reflection_agent.analyze_knowledge_gaps(
                        reflection_context
                    )
                else:
                    # Skip reflection, create a default result indicating sufficiency
                    reflection_result = ReflectionResult(
                        is_sufficient=True,
                        confidence_score=0.8,
                        knowledge_gaps=[],
                        follow_up_queries=[],
                        recommendations=[f"Reflection skipped: {trigger_reason}"],
                        reflection_insights=[],
                        comprehensive_report=f"Reflection analysis was skipped due to: {trigger_reason}",
                    )

                logger.info(
                    f"Iteration {current_iteration}: Sufficient={reflection_result.is_sufficient}, Confidence={reflection_result.confidence_score}, Gaps={len(reflection_result.knowledge_gaps)}"
                )

                # Check termination conditions
                if (
                    reflection_result.is_sufficient
                    or reflection_result.confidence_score >= sufficiency_threshold
                    or not reflection_result.follow_up_queries
                    or current_iteration >= max_iterations
                ):

                    logger.info(
                        f"Terminating iterative research: sufficient={reflection_result.is_sufficient}, confidence={reflection_result.confidence_score}, queries={len(reflection_result.follow_up_queries) if reflection_result.follow_up_queries else 0}"
                    )
                    break

                # Execute follow-up queries
                queries_to_execute = reflection_result.follow_up_queries[
                    :max_queries_per_iteration
                ]
                logger.info(
                    f"Executing {len(queries_to_execute)} follow-up queries (limited from {len(reflection_result.follow_up_queries)} total)..."
                )

                for i, query in enumerate(queries_to_execute):
                    try:
                        logger.info(
                            f"Executing follow-up query {i+1}: {query[:100]}..."
                        )

                        # Create a new research task for the follow-up query
                        follow_up_state = {
                            **state,
                            "messages": [{"role": "user", "content": query}],
                            "research_topic": query,
                            "is_follow_up_query": True,
                            "parent_iteration": current_iteration,
                        }

                        # Execute follow-up research using the same agent and tools
                        follow_up_result = await context_extension.execute_researcher_step_with_isolation(
                            follow_up_state, config, agent, "researcher"
                        )

                        # Extract findings from follow-up research
                        if (
                            hasattr(follow_up_result, "update")
                            and follow_up_result.update
                        ):
                            follow_up_observations = follow_up_result.update.get(
                                "observations", []
                            )
                            follow_up_messages = follow_up_result.update.get(
                                "messages", []
                            )

                            # Add follow-up findings to accumulated research
                            for obs in follow_up_observations:
                                if isinstance(obs, dict) and "content" in obs:
                                    all_research_findings.append(
                                        f"[Follow-up {current_iteration}.{i+1}] {obs['content']}"
                                    )
                                    all_observations.append(obs)
                                elif isinstance(obs, str):
                                    all_research_findings.append(
                                        f"[Follow-up {current_iteration}.{i+1}] {obs}"
                                    )
                                    all_observations.append(
                                        {"content": obs, "type": "follow_up_research"}
                                    )
                                else:
                                    content = str(obs)
                                    all_research_findings.append(
                                        f"[Follow-up {current_iteration}.{i+1}] {content}"
                                    )
                                    all_observations.append(
                                        {
                                            "content": content,
                                            "type": "follow_up_research",
                                        }
                                    )

                            for msg in follow_up_messages:
                                if isinstance(msg, dict) and "content" in msg:
                                    all_research_findings.append(
                                        f"[Follow-up {current_iteration}.{i+1}] {msg['content']}"
                                    )
                                elif hasattr(msg, "content"):
                                    all_research_findings.append(
                                        f"[Follow-up {current_iteration}.{i+1}] {msg.content}"
                                    )
                                else:
                                    all_research_findings.append(
                                        f"[Follow-up {current_iteration}.{i+1}] {str(msg)}"
                                    )

                        logger.info(f"Follow-up query {i+1} completed successfully")

                        # Add delay between queries within the same iteration
                        if i < len(queries_to_execute) - 1 and follow_up_delay > 0:
                            import asyncio

                            await asyncio.sleep(follow_up_delay)

                    except Exception as follow_up_error:
                        logger.error(f"Follow-up query {i+1} failed: {follow_up_error}")
                        continue

                # Add a small delay between iterations to prevent rate limiting
                import asyncio

                await asyncio.sleep(follow_up_delay)

            # Final reflection analysis with all accumulated findings
            logger.info(
                "Performing final reflection analysis with all accumulated findings..."
            )

            final_reflection_context = ReflectionContext(
                research_topic=state.get("research_topic", ""),
                completed_steps=completed_steps,
                execution_results=all_research_findings,
                observations=all_observations,
                total_steps=(
                    len(current_plan.steps)
                    if current_plan and hasattr(current_plan, "steps")
                    else 1
                ),
                current_step_index=len(completed_steps),
                locale=state.get("locale", "en-US"),
            )

            # Check if final reflection should be triggered
            should_trigger_final, final_trigger_reason, final_decision_factors = (
                reflection_integrator.should_trigger_reflection(
                    state=state, current_step=None, agent_name="researcher"
                )
            )

            logger.info(
                f"Final reflection trigger decision: {should_trigger_final}, reason: {final_trigger_reason}"
            )

            # Perform final reflection analysis only if triggered
            if should_trigger_final:
                final_reflection_result = await reflection_agent.analyze_knowledge_gaps(
                    final_reflection_context
                )
            else:
                # Skip final reflection, create a default result indicating sufficiency
                final_reflection_result = ReflectionResult(
                    is_sufficient=True,
                    confidence_score=0.8,
                    knowledge_gaps=[],
                    follow_up_queries=[],
                    recommendations=[
                        f"Final reflection skipped: {final_trigger_reason}"
                    ],
                    reflection_insights=[],
                    comprehensive_report=f"Final reflection analysis was skipped due to: {final_trigger_reason}",
                )

            # Integrate final reflection results into the isolation result
            if hasattr(isolation_result, "update") and isolation_result.update:
                # Update observations with all accumulated findings
                isolation_result.update["observations"] = all_observations

                # Only send reflection_insights to frontend if research is sufficient
                # This prevents sending incomplete or insufficient analysis results
                if final_reflection_result.is_sufficient:
                    isolation_result.update["reflection_insights"] = {
                        "comprehensive_report": (
                            final_reflection_result.comprehensive_report
                        ),
                        "knowledge_gaps": [
                            gap.to_dict() if hasattr(gap, "to_dict") else gap
                            for gap in final_reflection_result.knowledge_gaps
                        ],
                        "is_sufficient": final_reflection_result.is_sufficient,
                        "follow_up_queries": final_reflection_result.follow_up_queries,
                        "confidence_score": final_reflection_result.confidence_score,
                        "sufficiency_score": final_reflection_result.confidence_score,
                        "iterations_completed": current_iteration,
                        "total_research_findings": len(all_research_findings),
                    }
                    logger.info(
                        "Reflection insights sent to frontend - research is sufficient"
                    )
                else:
                    logger.info(
                        "Skipping reflection insights to frontend - research is insufficient (is_sufficient=False)"
                    )

                # If still insufficient after all iterations, add remaining follow-up suggestions
                if (
                    not final_reflection_result.is_sufficient
                    and final_reflection_result.follow_up_queries
                ):
                    isolation_result.update["suggested_follow_up"] = (
                        final_reflection_result.follow_up_queries
                    )
                    logger.info(
                        f"Final reflection identified {len(final_reflection_result.follow_up_queries)} additional follow-up queries"
                    )

                # Add comprehensive reflection metadata
                isolation_result.update["reflection_metadata"] = {
                    "reflection_applied": True,
                    "reflection_timestamp": time.time(),
                    "gaps_detected": len(final_reflection_result.knowledge_gaps),
                    "confidence_level": final_reflection_result.confidence_score,
                    "iterations_completed": current_iteration,
                    "max_iterations": max_iterations,
                    "sufficiency_threshold": sufficiency_threshold,
                    "final_sufficiency": final_reflection_result.is_sufficient,
                    "total_findings": len(all_research_findings),
                    "iterative_research_enabled": True,
                }

            logger.info(
                f"Iterative research completed - Final Sufficient: {final_reflection_result.is_sufficient}, Iterations: {current_iteration}, Total Findings: {len(all_research_findings)}, Final Confidence: {final_reflection_result.confidence_score}"
            )

        except Exception as e:
            logger.error(f"Enhanced reflection with iterative research failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Proceeding without reflection enhancement")

            # Add error metadata
            if hasattr(isolation_result, "update") and isolation_result.update:
                isolation_result.update["reflection_metadata"] = {
                    "reflection_applied": False,
                    "reflection_error": str(e),
                    "reflection_timestamp": time.time(),
                    "iterative_research_enabled": True,
                    "error_during_iteration": (
                        current_iteration if "current_iteration" in locals() else 0
                    ),
                }

    # Phase 5: Add reflection support methods to the result
    if hasattr(isolation_result, "update") and isolation_result.update:
        # Add reflection helper functions as metadata instead of methods
        # This avoids Pydantic __setattr__ issues with Command objects
        has_reflection_insights = "reflection_insights" in isolation_result.update
        reflection_helpers = {
            "has_reflection_insights": has_reflection_insights,
            "is_research_sufficient": (
                isolation_result.update.get("reflection_insights", {}).get(
                    "is_sufficient",
                    False,  # Default to False when no reflection insights
                )
                if has_reflection_insights
                else False  # No reflection insights means insufficient by default
            ),
            "reflection_insights": isolation_result.update.get(
                "reflection_insights", {}
            ),
        }

        # Store helper data in the update dict instead of as methods
        isolation_result.update["reflection_helpers"] = reflection_helpers

    return isolation_result


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
