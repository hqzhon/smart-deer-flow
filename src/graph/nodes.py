# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
import os
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


def check_and_truncate_messages(messages, max_messages=50, model_name="deepseek-chat", max_tokens=None):
    """Check and truncate messages to prevent excessive input length.
    
    Args:
        messages: List of messages to check
        max_messages: Maximum number of messages (fallback)
        model_name: Model name for token limit lookup
        max_tokens: Override token limit
    """
    if not messages:
        return messages
    
    # Try to use token-based truncation if content processor is available
    try:
        from src.utils.content_processor import ContentProcessor
        from src.config.configuration import Configuration
        
        # Get current configuration for model limits
        config = Configuration.get_current()
        if config and hasattr(config, 'model_token_limits'):
            processor = ContentProcessor(config.model_token_limits)
            limits = processor.get_model_limits(model_name)
            
            # Use provided max_tokens or calculate from model limits
            token_limit = max_tokens or int(limits.safe_input_limit * 0.7)  # Reserve 30% for other context
            
            # Calculate total tokens
            total_tokens = 0
            for msg in messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                total_tokens += processor.estimate_tokens(content)
            
            # If within limit, return as is
            if total_tokens <= token_limit:
                return messages
            
            # Token-based truncation: keep first and last messages, truncate middle
            if len(messages) <= 2:
                # If only 1-2 messages, truncate content of the longest one
                if len(messages) == 1:
                    content = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
                    if processor.estimate_tokens(content) > token_limit:
                        # Truncate content to fit
                        chunks = processor.smart_chunk_content(content, model_name)
                        if chunks:
                            truncated_content = chunks[0]  # Take first chunk
                            if hasattr(messages[0], 'content'):
                                messages[0].content = truncated_content + "\n\n[Content truncated to fit token limit]"
                return messages
            
            # For multiple messages, keep first and last, selectively keep middle ones
            first_msg = messages[0]
            last_msg = messages[-1]
            middle_msgs = messages[1:-1]
            
            # Calculate tokens for first and last messages
            first_tokens = processor.estimate_tokens(first_msg.content if hasattr(first_msg, 'content') else str(first_msg))
            last_tokens = processor.estimate_tokens(last_msg.content if hasattr(last_msg, 'content') else str(last_msg))
            
            remaining_tokens = token_limit - first_tokens - last_tokens
            
            if remaining_tokens <= 0:
                # If first and last messages exceed limit, truncate them
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
    
    # if enable smart chunking or summarization, process the content
    if configurable.enable_smart_chunking or configurable.enable_content_summarization:
        try:
            from src.utils.content_processor import ContentProcessor
            from src.llms.llm import get_llm_by_type
            
            processor = ContentProcessor(configurable.model_token_limits)
            
            # get the current llm model name
            current_llm = get_llm_by_type(AGENT_LLM_MAP.get("planner", "basic"))
            model_name = getattr(current_llm, 'model_name', getattr(current_llm, 'model', 'unknown'))
            
            # Check if the content exceeds the token limit
            if processor.estimate_tokens(raw_results) > processor.get_model_limits(model_name).safe_input_limit:
                logger.info("Search results exceed token limit, applying smart processing")
                
                if configurable.enable_content_summarization:
                    # Use smart summarization
                    processed_results = processor.summarize_content(
                        raw_results, current_llm, model_name, configurable.summary_type
                    )
                    logger.info(f"Search results summarized: {len(raw_results)} -> {len(processed_results)} characters")
                else:
                    # Use smart chunking
                    chunks = processor.smart_chunk_content(raw_results, model_name, configurable.chunk_strategy)
                    processed_results = chunks[0] if chunks else raw_results
                    logger.info(f"Search results chunked: {len(raw_results)} -> {len(processed_results)} characters")
                
                return {"background_investigation_results": processed_results}
            else:
                logger.info("Search results within token limit, no processing needed")
                
        except Exception as e:
            logger.error(f"Smart content processing failed: {e}, using original results")
    
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
            context="Generate the full plan."
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
            context="Generate research plan"
        )
        full_response = response.content if hasattr(response, 'content') else str(response)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
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
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 1:  # the plan_iterations is increased before this check
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    try:
        validated_plan = Plan.model_validate(new_plan)
    except Exception as e:
        logger.warning(f"Human feedback node execution error: Failed to parse Plan from completion {new_plan}. Got: {e}")
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
        context="Coordinating with customers"
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
        context="Generate the final report"
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
        logger.error(f"Parallel executor failed: {e.with_traceback}")
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
    """Execute a single step in parallel context."""
    logger.info(f"Executing step in parallel: {step.title} with {agent_type}")
    
    # Get completed steps for context with intelligent selection
    current_plan = state.get("current_plan")
    completed_steps = [s for s in current_plan.steps if s.execution_res and s.title != step.title]
    
    # Intelligent context management: limit and prioritize completed steps
    configurable = Configuration.from_runnable_config(config)
    max_context_steps = getattr(configurable, 'max_context_steps_parallel', 5)  # Limit context steps
    
    # Sort completed steps by relevance (you can implement more sophisticated relevance scoring)
    # For now, prioritize more recent steps and limit the number
    if len(completed_steps) > max_context_steps:
        # Take the most recent steps
        completed_steps = completed_steps[-max_context_steps:]
    
    # Format completed steps information with token awareness
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Existing Research Findings\n\n"
        
        # Try to use content processor for intelligent summarization
        try:
            from src.utils.content_processor import ContentProcessor
            processor = ContentProcessor(configurable.model_token_limits)
            
            for i, completed_step in enumerate(completed_steps):
                step_content = f"## Existing Finding {i + 1}: {completed_step.title}\n\n"
                step_result = completed_step.execution_res
                
                # Estimate tokens and summarize if too long
                if processor.estimate_tokens(step_result) > 1000:  # If step result is too long
                    # Summarize the step result
                    summary_prompt = processor.create_summary_prompt(step_result, "key_points")
                    try:
                        # Use basic model for summarization to save costs
                        from src.llms.llm import get_llm_by_type
                        basic_llm = get_llm_by_type("basic")
                        summary_result = basic_llm.invoke(summary_prompt)
                        step_result = summary_result.content if hasattr(summary_result, 'content') else str(summary_result)
                        step_result += "\n\n[This content has been automatically summarized to save tokens]"
                    except Exception as e:
                        logger.warning(f"Failed to summarize step result: {e}")
                        # Fallback: truncate to first 500 characters
                        step_result = step_result[:500] + "...\n\n[Content truncated]"
                
                step_content += f"<finding>\n{step_result}\n</finding>\n\n"
                completed_steps_info += step_content
                
        except Exception as e:
            logger.warning(f"Failed to use content processor for context optimization: {e}")
            # Fallback to simple formatting
            for i, completed_step in enumerate(completed_steps):
                completed_steps_info += f"## Existing Finding {i + 1}: {completed_step.title}\n\n"
                # Simple truncation for fallback
                result_content = completed_step.execution_res
                if len(result_content) > 1000:
                    result_content = result_content[:1000] + "...\n\n[Content truncated]"
                completed_steps_info += f"<finding>\n{result_content}\n</finding>\n\n"
    
    # Prepare agent input
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"{completed_steps_info}# Current Task\n\n## Title\n\n{step.title}\n\n## Description\n\n{step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
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
    
    # Fallback to default model name if not found
    if not model_name:
        model_name = "deepseek-chat"  # Default model name
    
    agent_input["messages"] = check_and_truncate_messages(
        agent_input["messages"], 
        model_name=model_name,
        max_messages=20  # Reduced from default 50 for parallel execution
    )
    
    result = await safe_llm_call_async(
        agent.ainvoke,
        input=agent_input,
        config={"recursion_limit": recursion_limit},
        operation_name=f"{agent_type} executor (parallel)",
        context=f"Parallel execution of step: {step.title}"
    )
    
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
    state: State, agent, agent_name: str
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
        recursion_limit = default_recursion_limit

    logger.info(f"Agent input: {agent_input}")
    
    result = await safe_llm_call_async(
        agent.ainvoke,
        input=agent_input, 
        config={"recursion_limit": recursion_limit},
        operation_name=f"{agent_name} executor",
        context=f"Execute step: {current_step.title}"
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
            return await _execute_agent_step(state, agent, agent_type)
    else:
        # Use default tools if no MCP servers are configured
        agent = create_agent(agent_type, agent_type, default_tools, agent_type)
        return await _execute_agent_step(state, agent, agent_type)


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
