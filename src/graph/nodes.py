# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
import os
import time
import traceback
from typing import Annotated, Literal, Optional, List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents import create_agent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call, safe_llm_call_async, handle_llm_errors
from src.prompts.planner_model import Plan, Step, StepType


def check_and_truncate_messages(messages, max_messages=50, model_name="deepseek-chat", max_tokens=None, aggressive_mode=False):
    """Check and truncate messages to prevent excessive input length with enhanced strategies.
    
    Args:
        messages: List of messages to check (can be strings or message objects)
        max_messages: Maximum number of messages (fallback)
        model_name: Model name for token limit lookup
        max_tokens: Override token limit
        aggressive_mode: Enable more aggressive truncation for tight budgets
    """
    if not messages:
        return messages
    
    # Convert string inputs to simple message objects for consistent processing
    processed_messages = []
    for msg in messages:
        if isinstance(msg, str):
            # Create a simple object with content attribute
            class SimpleMessage:
                def __init__(self, content):
                    self.content = content
                def __str__(self):
                    return self.content
            processed_messages.append(SimpleMessage(msg))
        else:
            processed_messages.append(msg)
    
    # Use processed_messages for the rest of the function
    messages = processed_messages
    
    # Try to use token-based truncation if content processor is available
    try:
        from src.utils.content_processor import ContentProcessor
        from src.config.configuration import Configuration
        from src.config.config_loader import config_loader
        
        # Get current configuration for model limits
        config = Configuration.get_current()
        model_token_limits = None
        
        if config and hasattr(config, 'model_token_limits') and config.model_token_limits:
            model_token_limits = config.model_token_limits
            logger.debug(f"Using model_token_limits from current config: {list(model_token_limits.keys())}")
        else:
            # Fallback: load directly from config file
            try:
                config_data = config_loader.load_config()
                # Build model_token_limits from BASIC_MODEL and REASONING_MODEL configurations
                model_token_limits = {}
                
                # Import ModelTokenLimits for proper object creation
                from src.utils.content_processor import ModelTokenLimits
                
                # Extract token limits from BASIC_MODEL
                basic_model = config_data.get('BASIC_MODEL', {})
                if basic_model.get('token_limits'):
                    basic_model_name = basic_model.get('model', 'deepseek-chat')
                    basic_limits = basic_model['token_limits']
                    model_token_limits[basic_model_name] = ModelTokenLimits(
                        input_limit=basic_limits.get('input_limit', 65536),
                        output_limit=basic_limits.get('output_limit', 8192),
                        context_window=basic_limits.get('context_window', 65536),
                        safety_margin=basic_limits.get('safety_margin', 0.8)
                    )
                    logger.debug(f"Added BASIC_MODEL token limits for {basic_model_name}: {model_token_limits[basic_model_name]}")
                
                # Extract token limits from REASONING_MODEL
                reasoning_model = config_data.get('REASONING_MODEL', {})
                if reasoning_model.get('token_limits'):
                    reasoning_model_name = reasoning_model.get('model', 'deepseek-reasoner')
                    reasoning_limits = reasoning_model['token_limits']
                    model_token_limits[reasoning_model_name] = ModelTokenLimits(
                        input_limit=reasoning_limits.get('input_limit', 65536),
                        output_limit=reasoning_limits.get('output_limit', 8192),
                        context_window=reasoning_limits.get('context_window', 65536),
                        safety_margin=reasoning_limits.get('safety_margin', 0.8)
                    )
                    logger.debug(f"Added REASONING_MODEL token limits for {reasoning_model_name}: {model_token_limits[reasoning_model_name]}")
                
                logger.info(f"Built model_token_limits from config file: {list(model_token_limits.keys())}")
            except Exception as e:
                logger.warning(f"Failed to load model_token_limits from config file: {e}")
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                model_token_limits = {}
        
        if model_token_limits:
            processor = ContentProcessor(model_token_limits)
            limits = processor.get_model_limits(model_name)
            
            # Use provided max_tokens or calculate from model limits
            if max_tokens is not None:
                token_limit = max_tokens
                # Enable aggressive mode for very tight budgets
                if max_tokens < 2000:
                    aggressive_mode = True
                    logger.info(f"Enabling aggressive mode due to tight budget: {max_tokens}")
            else:
                token_limit = int(limits.safe_input_limit * 0.7)  # Reserve 30% for other context
            
            # Calculate total tokens
            total_tokens = 0
            for msg in messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                total_tokens += processor.estimate_tokens(content)
            
            # If within limit, return as is
            if total_tokens <= token_limit:
                return messages
            
            logger.warning(f"Messages exceed token limit: {total_tokens} > {token_limit}, applying {'aggressive' if aggressive_mode else 'standard'} truncation")
            
            # Apply truncation strategy based on mode
            if aggressive_mode:
                # Aggressive mode: Keep only essential messages
                return _apply_aggressive_truncation(messages, processor, token_limit, model_name)
            
            # Standard token-based truncation: keep first and last messages, truncate middle
            if len(messages) <= 2:
                # If only 1-2 messages, truncate content of the longest one
                if len(messages) == 1:
                    content = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
                    if processor.estimate_tokens(content) > token_limit:
                        # Use aggressive chunking for extremely long content
                        if processor.estimate_tokens(content) > token_limit * 2:
                            logger.warning(f"Content extremely long ({processor.estimate_tokens(content)} tokens), using aggressive chunking")
                            chunks = processor.smart_chunk_content(content, model_name, "aggressive")
                        else:
                            chunks = processor.smart_chunk_content(content, model_name)
                        
                        if chunks:
                            truncated_content = chunks[0]  # Take first chunk
                            # Double-check the chunk size and further truncate if needed
                            if processor.estimate_tokens(truncated_content) > token_limit:
                                # Emergency truncation: use character-based limit
                                char_limit = int(token_limit * 3)  # Rough estimate: 1 token ≈ 3-4 characters
                                truncated_content = truncated_content[:char_limit]
                                logger.warning(f"Applied emergency character-based truncation to {char_limit} characters")
                            
                            if hasattr(messages[0], 'content'):
                                messages[0].content = truncated_content + "\n\n[Content truncated to fit token limit]"
                return messages
            
            # For multiple messages, keep first and last, selectively keep middle ones
            first_msg = messages[0]
            last_msg = messages[-1]
            middle_msgs = messages[1:-1]
            
            # Calculate tokens for first and last messages
            first_content = first_msg.content if hasattr(first_msg, 'content') else str(first_msg)
            last_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            first_tokens = processor.estimate_tokens(first_content)
            last_tokens = processor.estimate_tokens(last_content)
            
            # If first or last message is extremely long, truncate it
            if first_tokens > token_limit // 2:
                logger.warning(f"First message too long ({first_tokens} tokens), truncating")
                if processor.estimate_tokens(first_content) > token_limit:
                    chunks = processor.smart_chunk_content(first_content, model_name, "aggressive")
                    if chunks and hasattr(first_msg, 'content'):
                        first_msg.content = chunks[0] + "\n\n[First message truncated]"
                        first_tokens = processor.estimate_tokens(first_msg.content)
            
            if last_tokens > token_limit // 2:
                logger.warning(f"Last message too long ({last_tokens} tokens), truncating")
                if processor.estimate_tokens(last_content) > token_limit:
                    chunks = processor.smart_chunk_content(last_content, model_name, "aggressive")
                    if chunks and hasattr(last_msg, 'content'):
                        last_msg.content = chunks[0] + "\n\n[Last message truncated]"
                        last_tokens = processor.estimate_tokens(last_msg.content)
            
            remaining_tokens = token_limit - first_tokens - last_tokens
            
            if remaining_tokens <= 0:
                # If first and last messages still exceed limit, return minimal versions
                logger.warning("First and last messages exceed token limit even after truncation")
                return [first_msg, last_msg]
            
            # Select middle messages that fit in remaining tokens
            selected_middle = []
            current_tokens = 0
            
            # Prioritize more recent messages (from the end)
            for msg in reversed(middle_msgs):
                msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                msg_tokens = processor.estimate_tokens(msg_content)
                
                if current_tokens + msg_tokens <= remaining_tokens:
                    selected_middle.insert(0, msg)  # Insert at beginning to maintain order
                    current_tokens += msg_tokens
                else:
                    break
            
            # Add truncation notice if messages were removed
            result = [first_msg] + selected_middle + [last_msg]
            if len(result) < len(messages):
                truncation_notice = f"[Truncated {len(messages) - len(result)} messages to fit token limit]"
                if hasattr(result[-1], 'content'):
                    result[-1].content = truncation_notice + "\n\n" + result[-1].content
            
            return result
            
    except Exception as e:
        logger.warning(f"Token-based truncation failed, falling back to message count: {e}")
        logger.warning(f"Full traceback: {traceback.format_exc()}")
    
    # Fallback to original message count-based truncation
    if len(messages) > max_messages:
        keep_start = max_messages // 2
        keep_end = max_messages - keep_start
        truncated = messages[:keep_start] + messages[-keep_end:]
        return truncated
    
    return messages
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine

logger = logging.getLogger(__name__)


def _apply_aggressive_truncation(messages, processor, token_limit, model_name):
    """Apply aggressive truncation strategy for tight token budgets.
    
    Args:
        messages: List of messages to truncate
        processor: ContentProcessor instance
        token_limit: Maximum token limit
        model_name: Model name for processing
        
    Returns:
        Aggressively truncated list of messages
    """
    if not messages:
        return messages
    
    logger.info(f"Applying aggressive truncation with limit: {token_limit}")
    
    # Strategy: Keep only the most essential messages
    result_messages = []
    
    # Handle single message case first
    if len(messages) == 1:
        msg = messages[0]
        content = msg.content if hasattr(msg, 'content') else str(msg)
        msg_tokens = processor.estimate_tokens(content)
        
        if msg_tokens <= token_limit:
            return messages
        
        # Aggressive content truncation for single message
        logger.warning(f"Single message too long ({msg_tokens} tokens), applying aggressive truncation")
        max_chars = int(token_limit * 2.5)  # Conservative char estimate
        
        # Always truncate if we exceed token limit, regardless of char length
        if max_chars > 200:
            # Keep first 1/3 and last 2/3 of available space
            keep_start = max_chars // 3
            keep_end = max_chars - keep_start - 50  # Reserve for truncation marker
            
            if keep_end > 0 and len(content) > max_chars:
                truncated_content = (
                    content[:keep_start] + 
                    "\n\n...[content aggressively truncated]...\n\n" + 
                    content[-keep_end:]
                )
            else:
                # Simple truncation for shorter content or when keep_end <= 0
                truncated_content = content[:max_chars] + "\n\n[truncated]"
        else:
            truncated_content = content[:max_chars] + "\n\n[truncated]"
        
        if hasattr(msg, 'content'):
            msg.content = truncated_content
        
        logger.info(f"Single message truncated: {len(content)} -> {len(truncated_content)} chars")
        return [msg]
    
    # Always preserve system message if it exists and is reasonable size
    if messages and hasattr(messages[0], 'content'):
        first_content = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
        if 'system' in str(messages[0]).lower() or len(first_content) < 500:
            first_tokens = processor.estimate_tokens(first_content)
            if first_tokens < token_limit * 0.3:  # System message shouldn't take more than 30%
                result_messages.append(messages[0])
            else:
                # Truncate system message aggressively
                char_limit = int(token_limit * 0.3 * 2.5)  # Conservative char estimate
                truncated_content = first_content[:char_limit] + "...[system message truncated]"
                if hasattr(messages[0], 'content'):
                    messages[0].content = truncated_content
                result_messages.append(messages[0])
    
    # Calculate remaining budget
    used_tokens = sum(processor.estimate_tokens(
        msg.content if hasattr(msg, 'content') else str(msg)
    ) for msg in result_messages)
    remaining_budget = token_limit - used_tokens
    
    if remaining_budget <= 100:
        logger.warning(f"Very tight budget after system message: {remaining_budget}")
        return result_messages
    
    # Keep only the most recent user message and assistant response
    recent_messages = []
    
    # Look for the last user-assistant pair
    for i in range(len(messages) - 1, -1, -1):
        if len(recent_messages) >= 2:  # Already have a pair
            break
        
        msg = messages[i]
        if msg not in result_messages:  # Don't duplicate system message
            recent_messages.insert(0, msg)
    
    # Process recent messages with aggressive content truncation
    for msg in recent_messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        msg_tokens = processor.estimate_tokens(content)
        
        if msg_tokens <= remaining_budget:
            result_messages.append(msg)
            remaining_budget -= msg_tokens
        else:
            # Aggressive content truncation
            if remaining_budget > 50:
                # Smart truncation: keep beginning and end
                max_chars = int(remaining_budget * 2.5)  # Conservative estimate
                
                if len(content) > max_chars:
                    if max_chars > 100:
                        # Keep first 1/3 and last 2/3 of available space
                        keep_start = max_chars // 3
                        keep_end = max_chars - keep_start - 30  # Reserve for truncation marker
                        
                        if keep_end > 0:
                            truncated_content = (
                                content[:keep_start] + 
                                "...[content aggressively truncated]..." + 
                                content[-keep_end:]
                            )
                        else:
                            truncated_content = content[:max_chars] + "...[truncated]"
                    else:
                        truncated_content = content[:max_chars] + "...[truncated]"
                else:
                    truncated_content = content
                
                if hasattr(msg, 'content'):
                    msg.content = truncated_content
                result_messages.append(msg)
                break  # Only process one more message in aggressive mode
    
    # Final verification
    final_tokens = sum(processor.estimate_tokens(
        msg.content if hasattr(msg, 'content') else str(msg)
    ) for msg in result_messages)
    
    logger.info(f"Aggressive truncation complete: {len(result_messages)} messages, {final_tokens}/{token_limit} tokens")
    
    return result_messages


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
    configurable = Configuration.from_runnable_config(config)
    query = state.get("research_topic")
    background_investigation_results = None
    
    # get the background investigation results
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
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
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
        raw_results = json.dumps(background_investigation_results, ensure_ascii=False)
    
    # Smart content processing is always enabled (smart chunking, summarization, and smart filtering)
    if True:  # Always process content with smart features
        try:
            from src.utils.content_processor import ContentProcessor
            from src.llms.llm import get_llm_by_type
            
            processor = ContentProcessor(configurable.model_token_limits)
            
            # get the current llm model name
            model_name = 'unknown'
            try:
                current_llm = get_llm_by_type(AGENT_LLM_MAP.get("planner", "basic"))
                model_name = getattr(current_llm, 'model_name', getattr(current_llm, 'model', 'unknown'))
            except Exception as e:
                logger.warning(f"Failed to get LLM instance, trying to get model name from config: {e}")
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                try:
                    from src.config.config_loader import config_loader
                    config_data = config_loader.load_config()
                    agent_model = AGENT_LLM_MAP.get("planner", "basic")
                    
                    if agent_model == "basic":
                        basic_config = config_data.get("BASIC_MODEL", {})
                        model_name = basic_config.get("model", "deepseek-chat")
                    elif agent_model == "reasoning":
                        reasoning_config = config_data.get("REASONING_MODEL", {})
                        model_name = reasoning_config.get("model", "deepseek-reasoner")
                    
                    logger.debug(f"Retrieved model name from config for planner: {model_name}")
                    # Create a minimal LLM instance for processing
                    current_llm = get_llm_by_type("basic")  # Fallback to basic model
                except Exception as config_e:
                    logger.error(f"Failed to get model name from config: {config_e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    model_name = "deepseek-chat"  # Final fallback
                    current_llm = get_llm_by_type("basic")  # Fallback to basic model
            
            # Check if the content exceeds the token limit
            model_limits = processor.get_model_limits(model_name)
            current_tokens = processor.estimate_tokens(raw_results)
            token_limit_exceeded = current_tokens > model_limits.safe_input_limit
            
            # Use SearchResultFilter to determine if smart filtering should be enabled
            enable_smart_filtering = getattr(configurable, 'enable_smart_filtering', True)
            should_use_smart_filtering = False
            
            if enable_smart_filtering and isinstance(searched_content, list):
                from src.utils.search_result_filter import SearchResultFilter
                filter_instance = SearchResultFilter(processor)
                should_use_smart_filtering = filter_instance.should_enable_smart_filtering(searched_content, model_name)
                smart_filtering_threshold = filter_instance.get_smart_filtering_threshold(model_name)
            
            if token_limit_exceeded or should_use_smart_filtering:
                logger.info(f"Applying smart processing to search results (tokens: {current_tokens}, threshold: {smart_filtering_threshold if 'smart_filtering_threshold' in locals() else 'N/A'})")
                
                # Try smart filtering first if enabled and threshold is met
                if enable_smart_filtering and should_use_smart_filtering and isinstance(searched_content, list):
                    logger.info("Using smart filtering for search results (content exceeds 80% of input limit)")
                    processed_results = processor.process_search_results(
                        search_results=searched_content,
                        llm=current_llm,
                        model_name=model_name,
                        max_results=configurable.max_search_results,
                        query=query,
                        enable_smart_filtering=True
                    )
                    logger.info(f"Search results processed with smart filtering: {len(raw_results)} -> {len(processed_results)} characters")
                    return {"background_investigation_results": processed_results}
                
                # Fallback to traditional processing
                if configurable.enable_content_summarization:
                    # Use smart summarization
                    processed_results = processor.summarize_content(
                        raw_results, current_llm, model_name, configurable.summary_type
                    )
                    logger.info(f"Search results summarized: {len(raw_results)} -> {len(processed_results)} characters")
                else:
                    # Use smart chunking (always enabled with auto strategy)
                    chunks = processor.smart_chunk_content(raw_results, model_name, "auto")
                    processed_results = chunks[0] if chunks else raw_results
                    logger.info(f"Search results chunked: {len(raw_results)} -> {len(processed_results)} characters")
                
                return {"background_investigation_results": processed_results}
            else:
                logger.info("Search results within token limit, no processing needed")
                
        except Exception as e:
            logger.error(f"Smart content processing failed: {e}, using original results")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return {"background_investigation_results": raw_results}


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    messages = apply_prompt_template("planner", state, configurable)

    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    # Configure LLM based on settings
    use_structured_output = AGENT_LLM_MAP["planner"] == "basic" and not configurable.enable_deep_thinking
    
    if configurable.enable_deep_thinking:
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
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    if use_structured_output:
        response = safe_llm_call(
            llm.invoke, 
            messages,
            operation_name="Planner",
            context="Generate the full plan.",
            enable_context_evaluation=True
        )
        if hasattr(response, 'model_dump_json'):
            full_response = response.model_dump_json(indent=4, exclude_none=True)
        else:
            # if the response is not a structured output, return the default plan
            full_response = '{"steps": [{"title": "Research Task", "description": "Continue with general research due to content limitations."}]}'
    else:
        def stream_llm():
            response = llm.stream(messages)
            content = ""
            for chunk in response:
                content += chunk.content
            return type('StreamResponse', (), {'content': content})()
        
        response = safe_llm_call(
            stream_llm,
            operation_name="Planner streaming call",
            context="Generate research plan",
            enable_context_evaluation=True
        )
        full_response = response.content if hasattr(response, 'content') else str(response)
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
            logger.warning(f"Planner execution error: Failed to parse Plan from completion {curr_plan}. Got: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}") 
            if plan_iterations > 0:
                return Command(goto="reporter")
            else:
                return Command(goto="__end__")
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
            },
            goto="reporter",
        )
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
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
        logger.warning(f"Human feedback node execution error: Failed to parse Plan from completion {new_plan}. Got: {e}")
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
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("coordinator", state)
    response = safe_llm_call(
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke,
        messages,
        operation_name="Coordinator",
        context="Coordinating with customers",
        enable_context_evaluation=True
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
    configurable = Configuration.from_runnable_config(config)
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
    
    response = safe_llm_call(
        get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke,
        invoke_messages,
        operation_name="Reporter",
        context="Generate the final report",
        enable_context_evaluation=True
    )
    response_content = response.content if hasattr(response, 'content') else str(response)
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
    configurable = Configuration.from_runnable_config(config)
    enable_parallel = getattr(configurable, 'enable_parallel_execution', True)
    max_parallel_tasks = getattr(configurable, 'max_parallel_tasks', 3)
    
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
                "based on", "using", "from", "after", "following", 
                "according to", "based on", "using", "according to", "after", "reference"
            ]
            
            # Check if current step description mentions other step
            if any(keyword in step_desc_lower for keyword in dependency_keywords):
                if other_title_lower in step_desc_lower:
                    step_deps.add(other_step.title)
            
            # Processing steps typically depend on research steps
            if (step.step_type == StepType.PROCESSING and 
                other_step.step_type == StepType.RESEARCH):
                # Check for topic overlap
                step_words = set(step_title_lower.split())
                other_words = set(other_title_lower.split())
                if len(step_words.intersection(other_words)) >= 2:
                    step_deps.add(other_step.title)
        
        dependencies[step.title] = step_deps
    
    return dependencies


def _get_executable_steps(steps: List[Step], dependencies: Dict[str, Set[str]]) -> List[Step]:
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


async def _execute_steps_parallel(state: State, config: RunnableConfig, max_parallel_tasks: int) -> Command:
    """Execute steps in parallel using the ParallelExecutor with rate limiting."""
    logger.info(f"Starting parallel execution with max {max_parallel_tasks} concurrent tasks")
    
    from src.utils.parallel_executor import create_parallel_executor, ParallelTask, TaskPriority
    
    current_plan = state.get("current_plan")
    steps = current_plan.steps
    
    # Analyze dependencies
    dependencies = _analyze_step_dependencies(steps)
    logger.info(f"Step dependencies: {dependencies}")
    
    observations = state.get("observations", [])
    all_messages = []
    
    # Create parallel executor with rate limiting
    executor = create_parallel_executor(
        max_concurrent_tasks=max_parallel_tasks,
        enable_rate_limiting=True,
        enable_adaptive_scheduling=True
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
            dependencies=task_dependencies
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
        
        # Process results
        for task_id, task_result in results.items():
            step = next((s for s in steps if s.title == task_id), None)
            if not step:
                continue
                
            if task_result.status.value == "completed":
                logger.info(f"Step '{task_id}' completed successfully")
                if task_result.result and hasattr(task_result.result, 'content'):
                    observations.append(task_result.result.content)
                    # Get the agent type from the step
                    agent_type = "researcher" if step.step_type == StepType.RESEARCH else "coder"
                    all_messages.append(AIMessage(content=task_result.result.content, name=agent_type))
            else:
                logger.error(f"Step '{task_id}' failed with status: {task_result.status.value}")
                if task_result.error:
                    logger.error(f"Error details: {task_result.error}")
                # Ensure step has some result to prevent infinite loop
                if not step.execution_res:
                    error_msg = str(task_result.error) if task_result.error else "Execution failed"
                    step.execution_res = f"Execution failed: {error_msg}"
        
        # Log execution statistics
        stats = executor.get_stats()
        logger.info(f"Parallel execution stats: {stats}")
        
    except Exception as e:
        logger.error(f"Parallel executor failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Mark all unexecuted steps as failed
        for step in steps:
            if not step.execution_res:
                step.execution_res = f"Parallel Executor failed: {str(e)}"
    
    # Check if all steps are completed
    if all(step.execution_res for step in steps):
        logger.info("All steps completed via parallel execution")
        return Command(
            update={
                "messages": all_messages,
                "observations": observations,
            },
            goto="planner"
        )
    else:
        # Some steps still pending, continue
        return Command(
            update={
                "messages": all_messages,
                "observations": observations,
            },
            goto="research_team"
        )


async def _execute_single_step_parallel(state: State, config: RunnableConfig, step: Step, agent_type: str) -> Any:
    """Execute a single step in parallel context with proactive token budget management."""
    logger.info(f"Executing step in parallel: {step.title} with {agent_type}")
    
    # Get completed steps for context
    current_plan = state.get("current_plan")
    completed_steps = [s for s in current_plan.steps if s.execution_res and s.title != step.title]
    
    # Initialize configuration and processors
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize token budget manager for proactive management
    task_id = f"{agent_type}_{step.title}_{int(time.time())}"
    
    try:
        from src.utils.token_budget_manager import TokenBudgetManager, BudgetPriority
        from src.utils.content_processor import ContentProcessor
        
        content_processor = ContentProcessor(configurable.model_token_limits)
        budget_manager = TokenBudgetManager(content_processor, configurable)
        
        # Get model name for budget calculation
        agent_model = AGENT_LLM_MAP.get(agent_type, "basic")
        model_name = "deepseek-chat"  # Default fallback
        
        try:
            from src.config.config_loader import config_loader
            config_data = config_loader.load_config()
            
            if agent_model == "basic":
                basic_config = config_data.get("BASIC_MODEL", {})
                model_name = basic_config.get("model", "deepseek-chat")
            elif agent_model == "reasoning":
                reasoning_config = config_data.get("REASONING_MODEL", {})
                model_name = reasoning_config.get("model", "deepseek-reasoner")
        except Exception as e:
            logger.warning(f"Failed to get model name from config: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
        
        # Estimate parallel task count for budget allocation
        parallel_tasks = len([s for s in current_plan.steps if not s.execution_res])
        parallel_tasks = max(1, min(parallel_tasks, 3))  # Cap at 3 for resource management
        
        # Allocate token budget for this task
        task_allocation = budget_manager.allocate_task_budget(
            task_id=task_id,
            model_name=model_name,
            priority=BudgetPriority.HIGH,
            parallel_tasks=parallel_tasks
        )
        
        logger.info(f"Allocated {task_allocation.allocated_tokens} tokens for task {task_id}")
        
    except Exception as e:
        logger.warning(f"Token budget manager initialization failed: {e}, falling back to legacy management")
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        budget_manager = None
        task_allocation = None
    
    # Context management with budget-aware optimization
    try:
        from src.utils.advanced_context_manager import AdvancedContextManager, CompressionStrategy
        
        if budget_manager:
            # Get available context budget
            context_budget = budget_manager.get_context_budget(task_id)
            logger.info(f"Available context budget for task {task_id}: {context_budget} tokens")
        else:
            context_budget = None
        
        context_manager = AdvancedContextManager(configurable, content_processor)
        
        # Check if context sharing should be disabled
        disable_context_parallel = getattr(configurable, 'disable_context_parallel', False)
        if disable_context_parallel:
            logger.info("Context sharing disabled for parallel execution")
            completed_steps = []
        
        # Select compression strategy based on budget constraints
        strategy = CompressionStrategy.ADAPTIVE
        if disable_context_parallel:
            strategy = CompressionStrategy.NONE
        elif context_budget and context_budget < 1000:  # Very limited budget
            strategy = CompressionStrategy.HIERARCHICAL
            logger.info(f"Using hierarchical compression due to limited context budget: {context_budget}")
        elif len(completed_steps) > 3:
            strategy = CompressionStrategy.HIERARCHICAL
        elif len(completed_steps) > 1:
            strategy = CompressionStrategy.SLIDING_WINDOW
        
        # Optimize context using advanced manager with budget constraints
        current_task = f"Title: {step.title}\n\nDescription: {step.description}\n\nLocale: {state.get('locale', 'en-US')}"
        
        # If we have budget constraints, modify the context manager's max_context_ratio
        if budget_manager and context_budget:
            # Calculate what ratio of model limit this budget represents
            limits = content_processor.get_model_limits(model_name)
            budget_ratio = min(0.6, context_budget / limits.safe_input_limit)
            context_manager.max_context_ratio = budget_ratio
            logger.info(f"Adjusted context ratio to {budget_ratio:.2f} based on budget constraints")
        
        context_window = context_manager.optimize_context_for_parallel(
            completed_steps=completed_steps,
            current_task=current_task,
            model_name=model_name,
            strategy=strategy
        )
        
        # Format optimized context
        main_content = context_manager.format_context_window(context_window)
        
        # Log optimization statistics
        stats = context_manager.get_optimization_stats(context_window)
        logger.info(f"Context optimization stats: {stats['total_tokens']}/{stats['max_tokens']} tokens, "
                   f"compression: {stats['compression_ratio']:.2%}, strategy: {stats['strategy_used']}")
        
    except Exception as e:
        logger.error(f"Advanced context management failed: {e}, attempting recovery with conservative settings")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Instead of falling back to simple management, retry with more conservative settings
        try:
            # Create a new context manager with very conservative settings
            context_manager = AdvancedContextManager(configurable, content_processor)
            context_manager.max_context_ratio = 0.3  # Very conservative ratio
            context_manager.sliding_window_size = 2   # Minimal window
            context_manager.compression_threshold = 0.5  # Early compression
            
            # Force hierarchical compression for maximum token reduction
            strategy = CompressionStrategy.HIERARCHICAL
            
            # Limit completed steps more aggressively
            if len(completed_steps) > 2:
                completed_steps = completed_steps[-2:]  # Only keep last 2 steps
            
            logger.info("Retrying context optimization with conservative settings")
            
            context_window = context_manager.optimize_context_for_parallel(
                completed_steps=completed_steps,
                current_task=current_task,
                model_name=model_name,
                strategy=strategy
            )
            
            # Format optimized context
            main_content = context_manager.format_context_window(context_window)
            
            # Log recovery statistics
            stats = context_manager.get_optimization_stats(context_window)
            logger.info(f"Recovery context stats: {stats['total_tokens']}/{stats['max_tokens']} tokens, "
                       f"compression: {stats['compression_ratio']:.2%}, strategy: {stats['strategy_used']}")
            
        except Exception as recovery_error:
            logger.error(f"Context management recovery also failed: {recovery_error}")
            logger.error(f"Recovery traceback: {traceback.format_exc()}")
            
            # Last resort: use minimal context with only current task
            logger.warning("Using minimal context with current task only")
            main_content = f"# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
    
    agent_input = {
        "messages": [
            HumanMessage(content=main_content)
        ]
    }
    
    # Add agent-specific configurations
    if agent_type == "researcher":
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"
            
            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info + "\n\n" + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )
        
        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )
    
    # Create agent with appropriate tools
    configurable = Configuration.from_runnable_config(config)
    
    if agent_type == "researcher":
        tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)
    elif agent_type == "coder":
        tools = [python_repl_tool]
    else:
        tools = []
    
    agent = create_agent(agent_type, agent_type, tools, agent_type)
    
    # Execute agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        recursion_limit = int(env_value_str) if int(env_value_str) > 0 else default_recursion_limit
    except ValueError:
        recursion_limit = default_recursion_limit
    
    # Add input length check with token-based truncation
    # Get model name from agent configuration
    agent_model = AGENT_LLM_MAP.get(agent_type, "basic")
    model_name = None
    
    try:
        if agent_model == "basic" and configurable.basic_model:
            model_name = getattr(configurable.basic_model, 'model_name', None)
        elif agent_model == "reasoning" and configurable.reasoning_model:
            model_name = getattr(configurable.reasoning_model, 'model_name', None)
    except AttributeError:
        pass
    
    # If model instance is not available, get model name from configuration
    if not model_name:
        try:
            from src.config.config_loader import config_loader
            config_data = config_loader.load_config()
            
            if agent_model == "basic":
                basic_config = config_data.get("BASIC_MODEL", {})
                model_name = basic_config.get("model")
            elif agent_model == "reasoning":
                reasoning_config = config_data.get("REASONING_MODEL", {})
                model_name = reasoning_config.get("model")
                
            logger.debug(f"Retrieved model name from config: {model_name} for agent_model: {agent_model}")
        except Exception as e:
            logger.warning(f"Failed to get model name from config: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
    
    # Final fallback to default model name if still not found
    if not model_name:
        model_name = "deepseek-chat"  # Default model name
        logger.warning(f"Using default model name: {model_name} for agent_model: {agent_model}")
    
    # Apply budget-aware message optimization
    if budget_manager:
        try:
            # Use budget manager to optimize messages
            agent_input["messages"] = budget_manager.optimize_message_for_budget(
                task_id=task_id,
                messages=agent_input["messages"],
                model_name=model_name
            )
            
            # Validate final message budget
            is_valid, estimated_tokens, available_tokens = budget_manager.validate_message_budget(
                task_id, agent_input["messages"], model_name
            )
            
            if not is_valid:
                logger.error(f"Message budget validation failed for task {task_id}: "
                           f"{estimated_tokens} > {available_tokens}")
            else:
                logger.info(f"Message budget validated for task {task_id}: {estimated_tokens}/{available_tokens} tokens")
                
        except Exception as e:
            logger.warning(f"Budget-aware message optimization failed: {e}, falling back to legacy truncation")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            # Fallback to legacy truncation with aggressive mode
            agent_input["messages"] = check_and_truncate_messages(
                agent_input["messages"], 
                model_name=model_name,
                max_messages=10,  # Further reduced for parallel execution
                max_tokens=int(content_processor.get_model_limits(model_name).safe_input_limit * 0.6),
                aggressive_mode=True  # Enable aggressive mode for parallel execution
            )
    else:
        # Legacy truncation when budget manager is not available
        agent_input["messages"] = check_and_truncate_messages(
            agent_input["messages"], 
            model_name=model_name,
            max_messages=10,  # Further reduced for parallel execution
            max_tokens=int(content_processor.get_model_limits(model_name).safe_input_limit * 0.6),
            aggressive_mode=True  # Enable aggressive mode for parallel execution
        )
    
    # Execute agent with budget tracking
    try:
        result = await safe_llm_call_async(
            agent.ainvoke,
            input=agent_input,
            config={"recursion_limit": recursion_limit},
            operation_name=f"{agent_type} executor (parallel)",
            context=f"Parallel execution of step: {step.title}",
            enable_smart_processing=True,
            enable_context_evaluation=True,
            max_retries=2  # Reduce retries for parallel execution
        )
        
        # Update token usage tracking
        if budget_manager:
            try:
                # Estimate tokens used in the request
                input_tokens = budget_manager.estimate_message_tokens(agent_input["messages"])
                
                # Estimate output tokens (rough approximation)
                if hasattr(result, 'get') and "messages" in result:
                    output_content = result["messages"][-1].content
                else:
                    output_content = result.content if hasattr(result, 'content') else str(result)
                
                output_tokens = content_processor.estimate_tokens(output_content)
                total_used = input_tokens + output_tokens
                
                budget_manager.update_token_usage(task_id, total_used)
                
                # Log budget utilization
                stats = budget_manager.get_budget_stats()
                logger.info(f"Token usage updated for task {task_id}: {total_used} tokens. "
                           f"Overall utilization: {stats['utilization_rate']:.2%}")
                
            except Exception as e:
                logger.warning(f"Failed to update token usage tracking: {e}")
                logger.warning(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        # Always release budget allocation when task completes
        if budget_manager:
            budget_manager.release_task_budget(task_id)
    
    # Process result
    if hasattr(result, 'get') and "messages" in result:
        response_content = result["messages"][-1].content
    else:
        response_content = result.content if hasattr(result, 'content') else str(result)
    
    # Update step with result
    step.execution_res = response_content
    logger.info(f"Parallel step '{step.title}' completed by {agent_type}")
    
    return type('StepResult', (), {'content': response_content})()


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
    """Helper function to execute a step using the specified agent."""
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
            completed_steps.append(step)

    if not current_step:
        logger.warning("No unexecuted step found")
        return Command(goto="research_team")

    logger.info(f"Executing step: {current_step.title}, agent: {agent_name}")

    # Check if context sharing should be disabled (apply parallel execution settings to serial execution too)
    configurable = Configuration.from_runnable_config(config)
    disable_context_parallel = getattr(configurable, 'disable_context_parallel', False)
    max_context_steps_parallel = getattr(configurable, 'max_context_steps_parallel', 1)
    
    # Apply context limitations
    if disable_context_parallel:
        logger.info("Context sharing disabled - no historical context will be included")
        completed_steps = []
    elif max_context_steps_parallel > 0 and len(completed_steps) > max_context_steps_parallel:
        # Limit the number of context steps
        completed_steps = completed_steps[-max_context_steps_parallel:]
        logger.info(f"Limited context to last {max_context_steps_parallel} steps")
    
    # Format completed steps information
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Existing Research Findings\n\n"
        for i, step in enumerate(completed_steps):
            completed_steps_info += f"## Existing Finding {i + 1}: {step.title}\n\n"
            completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    # Prepare the input for the agent with completed steps info
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"{completed_steps_info}# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }

    # Add citation reminder for researcher agent
    if agent_name == "researcher":
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

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

    # Get model name for token truncation
    model_name = None
    
    try:
        # Try to get model name from LLM instance
        agent_model = AGENT_LLM_MAP.get(agent_name, "basic")
        current_llm = get_llm_by_type(agent_model)
        model_name = getattr(current_llm, 'model_name', getattr(current_llm, 'model', None))
    except Exception as e:
        logger.warning(f"Failed to get LLM instance for {agent_name}, trying to get model name from config: {e}")
        logger.warning(f"Full traceback: {traceback.format_exc()}")
        try:
            from src.config.config_loader import config_loader
            config_data = config_loader.load_config()
            agent_model = AGENT_LLM_MAP.get(agent_name, "basic")
            
            if agent_model == "basic":
                basic_config = config_data.get("BASIC_MODEL", {})
                model_name = basic_config.get("model", "deepseek-chat")
            elif agent_model == "reasoning":
                reasoning_config = config_data.get("REASONING_MODEL", {})
                model_name = reasoning_config.get("model", "deepseek-reasoner")
            
            logger.debug(f"Retrieved model name from config for {agent_name}: {model_name}")
        except Exception as config_e:
            logger.error(f"Failed to get model name from config: {config_e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            model_name = "deepseek-chat"  # Final fallback
    
    # Final fallback to default model name if still not found
    if not model_name:
        model_name = "deepseek-chat"  # Default model name
        logger.warning(f"Using default model name: {model_name} for agent: {agent_name}")
    
    # Apply token truncation to prevent context length exceeded errors
    # Get token limits from configuration
    try:
        from src.config.config_loader import config_loader
        config_data = config_loader.load_config()
        
        # Determine which model config to use
        agent_model = AGENT_LLM_MAP.get(agent_name, "basic")
        if agent_model == "basic":
            model_config = config_data.get("BASIC_MODEL", {})
        elif agent_model == "reasoning":
            model_config = config_data.get("REASONING_MODEL", {})
        else:
            model_config = config_data.get("BASIC_MODEL", {})
        
        # Get token limits with safety margin
        token_limits = model_config.get("token_limits", {})
        input_limit = token_limits.get("input_limit", 65536)
        safety_margin = model_config.get("safety_margin", 0.8)
        max_tokens = int(input_limit * safety_margin * 0.6)  # Reserve 40% for response and other context
        
        logger.info(f"Using token limit {max_tokens} for {agent_name} (model: {model_name})")
        
        agent_input["messages"] = check_and_truncate_messages(
            agent_input["messages"], 
            model_name=model_name,
            max_messages=20,  # Reduced from default 50 for better performance
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to get token limits for truncation: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Fallback to conservative truncation
        agent_input["messages"] = check_and_truncate_messages(
            agent_input["messages"], 
            model_name=model_name,
            max_messages=10,  # Very conservative fallback
            max_tokens=20000  # Conservative token limit
        )
    
    logger.info(f"Agent input (after truncation): {len(agent_input['messages'])} messages")
    
    result = await safe_llm_call_async(
        agent.ainvoke,
        input=agent_input, 
        config={"recursion_limit": recursion_limit},
        operation_name=f"{agent_name} executor",
        context=f"Execute step: {current_step.title}",
        enable_smart_processing=True,
        enable_context_evaluation=True,
        max_retries=3
    )
    
    # Process the result
    if hasattr(result, 'get') and "messages" in result:
        response_content = result["messages"][-1].content
    else:
        # If the result is not a message, use the content directly
        response_content = result.content if hasattr(result, 'content') else str(result)
    
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")
    
    # Update the step with the execution result
    current_step.execution_res = response_content
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": observations + [response_content],
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
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}

    # Extract MCP server configuration for this agent type
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name

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
            agent = create_agent(agent_type, agent_type, loaded_tools, agent_type)
            return await _execute_agent_step(state, config, agent, agent_type)
    else:
        # Use default tools if no MCP servers are configured
        agent = create_agent(agent_type, agent_type, default_tools, agent_type)
        return await _execute_agent_step(state, config, agent, agent_type)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")
    configurable = Configuration.from_runnable_config(config)
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
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
