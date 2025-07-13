"""Researcher Context Isolator - Phase 1 Implementation

This module provides context isolation for researcher nodes to prevent
context accumulation issues in parallel execution scenarios.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from ..context.execution_context_manager import ExecutionContextManager, ContextConfig
from src.graph.types import State

logger = logging.getLogger(__name__)


@dataclass
class ResearcherContextConfig:
    """Configuration for researcher context isolation."""
    max_context_steps: int = 2  # Limit historical context for researchers
    max_step_content_length: int = 1500  # Shorter content for focused research
    max_observations_length: int = 8000  # Moderate observation limit
    enable_content_deduplication: bool = True
    enable_smart_truncation: bool = True
    isolation_level: str = "moderate"  # "minimal", "moderate", "aggressive"


class ResearcherContextIsolator:
    """Context isolator specifically designed for researcher nodes.
    
    This class extends ExecutionContextManager with researcher-specific
    optimizations to prevent context accumulation while maintaining
    research quality.
    """
    
    def __init__(self, config: Optional[ResearcherContextConfig] = None):
        """Initialize the researcher context isolator.
        
        Args:
            config: Configuration for context isolation. Uses defaults if None.
        """
        self.config = config or ResearcherContextConfig()
        
        # Convert to ExecutionContextManager config
        context_config = ContextConfig(
            max_context_steps=self.config.max_context_steps,
            max_step_content_length=self.config.max_step_content_length,
            max_observations_length=self.config.max_observations_length,
            enable_content_deduplication=self.config.enable_content_deduplication,
            enable_smart_truncation=self.config.enable_smart_truncation
        )
        
        self.context_manager = ExecutionContextManager(context_config)
        
        self.context_manager = ExecutionContextManager(context_config)
        logger.info(f"ResearcherContextIsolator initialized with isolation level: {self.config.isolation_level}")
    
    def prepare_isolated_context(
        self,
        state_or_steps: Union[State, List[Dict[str, Any]]],
        current_step: Optional[Dict[str, str]] = None,
        agent_name: str = "researcher"
    ) -> Union[State, Tuple[List[Dict[str, Any]], str]]:
        """Prepare isolated context for researcher execution.
        
        This method applies researcher-specific context isolation strategies
        to prevent context accumulation while preserving essential information.
        
        Args:
            state_or_steps: Either a State object or list of completed step dictionaries
            current_step: Current step dictionary with 'step' and 'description' keys (optional for State input)
            agent_name: Name of the agent (default: "researcher")
            
        Returns:
            State object if State was provided as input, otherwise tuple of (optimized_steps, context_info_string)
        """
        # Handle State object input
        if isinstance(state_or_steps, dict) and 'messages' in state_or_steps:
            return self._prepare_isolated_context_from_state(state_or_steps, agent_name)
        
        # Handle legacy list input
        completed_steps = state_or_steps
        if current_step is None:
            current_step = {"step": "current", "description": "Current step"}
            
        logger.debug(f"Preparing isolated context for {agent_name} with {len(completed_steps)} completed steps")
        
        # Apply isolation level-specific filtering
        filtered_steps = self._apply_isolation_filtering(completed_steps)
        
        # Use base context manager for core optimization
        optimized_steps, context_info = self.context_manager.prepare_context_for_execution(
            filtered_steps, current_step, agent_name
        )
        
        # Apply researcher-specific context enhancement
        enhanced_context_info = self._enhance_researcher_context(context_info, current_step)
        
        logger.debug(f"Context isolation completed: {len(filtered_steps)} -> {len(optimized_steps)} steps")
        return optimized_steps, enhanced_context_info
    
    def manage_isolated_observations(
        self,
        observations: List[str],
        new_observation: str
    ) -> List[str]:
        """Manage observations with researcher-specific isolation.
        
        Args:
            observations: Existing observations list
            new_observation: New observation to add
            
        Returns:
            Optimized observations list
        """
        all_observations = observations + [new_observation]
        
        # Use advanced observation management with researcher-specific settings
        optimization_level = self._get_optimization_level()
        
        optimized_observations = self.context_manager.manage_observations_advanced(
            all_observations,
            optimization_level=optimization_level
        )
        
        logger.debug(
            f"Observation isolation: {len(all_observations)} -> {len(optimized_observations)} observations"
        )
        
        return optimized_observations
    
    def _prepare_isolated_context_from_state(self, state: State, agent_name: str = "researcher") -> State:
        """Prepare isolated context from State object.
        
        Args:
            state: Input State object
            agent_name: Name of the agent
            
        Returns:
            Optimized State object
        """
        # Extract step history from state (if it exists)
        step_history = state.get('step_history', [])
        
        # Apply isolation filtering
        filtered_steps = self._apply_isolation_filtering(step_history)
        
        # Apply content truncation to ensure length limits
        filtered_steps = self._truncate_step_content(filtered_steps)
        
        # Create a new State object with the same attributes
        optimized_state = {
            'locale': state.get('locale', 'en-US'),
            'research_topic': state.get('research_topic', ''),
            'observations': state.get('observations', []),
            'resources': state.get('resources', []),
            'plan_iterations': state.get('plan_iterations', 0),
            'current_plan': state.get('current_plan', None),
            'final_report': state.get('final_report', ''),
            'auto_accepted_plan': state.get('auto_accepted_plan', False),
            'enable_background_investigation': state.get('enable_background_investigation', True),
            'background_investigation_results': state.get('background_investigation_results', None),
            'enable_collaboration': state.get('enable_collaboration', True),
            'collaboration_systems': state.get('collaboration_systems', None)
        }
        
        # Copy messages if they exist
        if 'messages' in state:
            optimized_state['messages'] = state['messages']
        
        # Add step history as a dynamic attribute
        optimized_state['step_history'] = filtered_steps
        
        # Optimize observations if they exist
        if optimized_state.get('observations'):
            optimization_level = self._get_optimization_level()
            optimized_observations = self.context_manager.manage_observations_advanced(
                optimized_state['observations'],
                optimization_level=optimization_level
            )
            optimized_state['observations'] = optimized_observations
        
        # Optimize messages if they exist and are too long
        if 'messages' in optimized_state and optimized_state['messages']:
            # Limit messages based on configuration
            max_messages = self.config.max_context_steps * 2
            if len(optimized_state['messages']) > max_messages:
                optimized_state['messages'] = optimized_state['messages'][-max_messages:]
        
        logger.debug(f"State isolation completed: {len(step_history)} -> {len(filtered_steps)} steps")
        return optimized_state
    
    def _apply_isolation_filtering(self, completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply isolation level-specific filtering to completed steps.
        
        Args:
            completed_steps: List of completed step dictionaries
            
        Returns:
            Filtered steps based on isolation level
        """
        if self.config.isolation_level == "minimal":
            # Keep more context for minimal isolation
            return completed_steps[-self.config.max_context_steps * 2:] if completed_steps else []
        elif self.config.isolation_level == "aggressive":
            # Keep only the most recent step for aggressive isolation
            return completed_steps[-1:] if completed_steps else []
        else:  # moderate
            # Use configured max_context_steps
            return completed_steps[-self.config.max_context_steps:] if completed_steps else []
    
    def _enhance_researcher_context(self, context_info: str, current_step: Dict[str, str]) -> str:
        """Enhance context information with researcher-specific guidance.
        
        Args:
            context_info: Base context information string
            current_step: Current step dictionary
            
        Returns:
            Enhanced context information
        """
        researcher_guidance = (
            "\n\n## Research Guidelines\n\n"
            "- Focus on the current research task without being influenced by previous unrelated research\n"
            "- Prioritize accuracy and relevance over comprehensiveness\n"
            "- Use appropriate tools for information gathering and verification\n"
            "- Maintain clear source attribution in your research findings"
        )
        
        return context_info + researcher_guidance
    
    def _get_optimization_level(self) -> str:
        """Get optimization level for observation management based on isolation level.
        
        Returns:
            Optimization level string for ExecutionContextManager
        """
        isolation_to_optimization = {
            "minimal": "conservative",
            "moderate": "balanced", 
            "aggressive": "aggressive"
        }
        return isolation_to_optimization.get(self.config.isolation_level, "balanced")
    
    def _truncate_step_content(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate step content to prevent memory overflow.
        
        Args:
            steps: List of step dictionaries
            
        Returns:
            Steps with truncated content
        """
        truncated_steps = []
        
        for step in steps:
            execution_res = step.get('execution_res', '')
            
            if len(execution_res) > self.config.max_step_content_length:
                # Simple truncation to max length
                truncated_content = execution_res[:self.config.max_step_content_length] + "...[截断]"
                
                modified_step = step.copy()
                modified_step['execution_res'] = truncated_content
                truncated_steps.append(modified_step)
            else:
                truncated_steps.append(step)
                
        return truncated_steps
    
    def capture_local_activity(self, activity_type: str, content: Any):
        """Capture local activity without propagating to global state.
        
        Args:
            activity_type: Type of activity (search, analysis, thought, message)
            content: Activity content
        """
        if activity_type == "search":
            if hasattr(self, 'isolated_search_results'):
                self.isolated_search_results.append(content)
            else:
                self.isolated_search_results = [content]
        elif activity_type == "analysis":
            if hasattr(self, 'isolated_analysis_steps'):
                self.isolated_analysis_steps.append(str(content))
            else:
                self.isolated_analysis_steps = [str(content)]
        elif activity_type == "thought":
            if hasattr(self, 'isolated_intermediate_thoughts'):
                self.isolated_intermediate_thoughts.append(str(content))
            else:
                self.isolated_intermediate_thoughts = [str(content)]
        elif activity_type == "message":
            if hasattr(self, 'isolated_messages'):
                self.isolated_messages.append(str(content))
            else:
                self.isolated_messages = [str(content)]
    
    def generate_refined_output(self) -> str:
        """Generate refined output for passing to global state.
        
        Returns:
            Refined output string containing key findings
        """
        key_findings = []
        
        # Extract key search results
        if hasattr(self, 'isolated_search_results') and self.isolated_search_results:
            search_summary = f"Search findings: {len(self.isolated_search_results)} results processed"
            key_findings.append(search_summary)
        
        # Extract key analysis steps
        if hasattr(self, 'isolated_analysis_steps') and self.isolated_analysis_steps:
            analysis_summary = f"Analysis completed: {len(self.isolated_analysis_steps)} steps"
            key_findings.append(analysis_summary)
        
        # Extract key thoughts
        if hasattr(self, 'isolated_intermediate_thoughts') and self.isolated_intermediate_thoughts:
            thought_summary = f"Research insights: {len(self.isolated_intermediate_thoughts)} thoughts captured"
            key_findings.append(thought_summary)
        
        # Extract messages
        if hasattr(self, 'isolated_messages') and self.isolated_messages:
            message_summary = f"Messages processed: {len(self.isolated_messages)} items"
            key_findings.append(message_summary)
        
        if not key_findings:
            return "Research session completed with no specific findings."
        
        return "\n".join(key_findings)
    
    def estimate_context_size(self) -> int:
        """Estimate the current context size in tokens.
        
        Returns:
            Estimated token count
        """
        total_size = 0
        
        # Estimate search results size
        if hasattr(self, 'isolated_search_results'):
            total_size += sum(len(str(result)) for result in self.isolated_search_results)
        
        # Estimate analysis steps size
        if hasattr(self, 'isolated_analysis_steps'):
            total_size += sum(len(step) for step in self.isolated_analysis_steps)
        
        # Estimate thoughts size
        if hasattr(self, 'isolated_intermediate_thoughts'):
            total_size += sum(len(thought) for thought in self.isolated_intermediate_thoughts)
        
        # Estimate messages size
        if hasattr(self, 'isolated_messages'):
            total_size += sum(len(msg) for msg in self.isolated_messages)
        
        # Rough token estimation (1 token ≈ 4 characters)
        return total_size // 4