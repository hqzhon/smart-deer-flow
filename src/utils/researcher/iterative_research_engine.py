"""Iterative Research Engine Module - Manages research iteration logic and termination conditions"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class IterativeResearchEngine:
    """Iterative Research Engine - Manages research iteration logic and termination conditions"""

    def __init__(self, unified_config: Any):
        """Initialize iterative research engine

        Args:
            unified_config: Unified configuration object
        """
        self.unified_config = unified_config
        self.iteration_count = 0
        self.start_time = None
        self.research_history = []

        logger.info("IterativeResearchEngine initialized")

    def start_research_session(self) -> None:
        """Start research session"""
        self.start_time = datetime.now()
        self.iteration_count = 0
        self.research_history = []

        logger.info(f"Research session started at {self.start_time}")

    def increment_iteration(self) -> int:
        """Increment iteration count

        Returns:
            Current iteration count
        """
        self.iteration_count += 1
        logger.info(f"Research iteration incremented to {self.iteration_count}")
        return self.iteration_count

    def add_research_record(
        self, query: str, result: Any, reflection_result: Any = None
    ) -> None:
        """Add research record

        Args:
            query: Query content
            result: Research result
            reflection_result: Reflection result (optional)
        """
        record = {
            "iteration": self.iteration_count,
            "timestamp": datetime.now(),
            "query": query,
            "result": result,
            "reflection_result": reflection_result,
        }

        self.research_history.append(record)
        logger.debug(f"Research record added for iteration {self.iteration_count}")

    def check_termination_conditions(
        self, state: Dict[str, Any], reflection_result: Any = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check iterative research termination conditions

        Args:
            state: Current state
            reflection_result: Reflection result (optional)

        Returns:
            (should_terminate, termination_reason, decision_factors) tuple
        """
        decision_factors = {
            "current_iteration": self.iteration_count,
            "max_iterations": self.unified_config.max_follow_up_iterations,
            "elapsed_time": self._get_elapsed_time(),
            "max_time": getattr(self.unified_config, "max_research_time_minutes", 30),
            "has_reflection_result": reflection_result is not None,
        }

        # Check maximum iterations
        if self.iteration_count >= self.unified_config.max_follow_up_iterations:
            reason = f"Reached maximum iterations ({self.unified_config.max_follow_up_iterations})"
            logger.info(f"Termination condition met: {reason}")
            return True, reason, decision_factors

        # Check time limit
        elapsed_minutes = self._get_elapsed_time()
        max_time = getattr(self.unified_config, "max_research_time_minutes", 30)
        if elapsed_minutes >= max_time:
            reason = f"Reached time limit ({max_time} minutes)"
            logger.info(f"Termination condition met: {reason}")
            return True, reason, decision_factors

        # Check if reflection result indicates research is sufficient
        if reflection_result and hasattr(reflection_result, "is_sufficient"):
            if reflection_result.is_sufficient:
                reason = "Reflection indicates research is sufficient"
                logger.info(f"Termination condition met: {reason}")
                decision_factors["reflection_sufficient"] = True
                decision_factors["confidence_score"] = getattr(
                    reflection_result, "confidence_score", None
                )
                return True, reason, decision_factors
            else:
                decision_factors["reflection_sufficient"] = False
                decision_factors["knowledge_gaps"] = (
                    1
                    if getattr(reflection_result, "primary_knowledge_gap", None)
                    else 0
                )

        # Check if there is a valid follow-up query
        primary_follow_up_query = state.get("primary_follow_up_query")
        if not primary_follow_up_query:
            reason = "No follow-up query available"
            logger.info(f"Termination condition met: {reason}")
            decision_factors["follow_up_queries_count"] = 0
            return True, reason, decision_factors

        decision_factors["follow_up_queries_count"] = 1

        # Continue research
        logger.debug(
            f"Continuing research - iteration {self.iteration_count}/{self.unified_config.max_follow_up_iterations}"
        )
        return False, "Continue research", decision_factors

    def generate_follow_up_queries(
        self, reflection_result: Any, state: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up query based on reflection result

        Args:
            reflection_result: Reflection result
            state: Current state

        Returns:
            List containing single follow-up query for backward compatibility
        """
        if not reflection_result or not hasattr(
            reflection_result, "primary_knowledge_gap"
        ):
            logger.warning("No reflection result or primary knowledge gap available")
            return []

        primary_gap = reflection_result.primary_knowledge_gap
        primary_query = reflection_result.primary_follow_up_query

        if not primary_gap:
            logger.info("No primary knowledge gap identified")
            return []

        # Generate single follow-up query based on primary gap and query
        if primary_query:
            follow_up_query = primary_query
        else:
            # If no primary query, generate one based on the gap
            follow_up_query = f"Research more about: {primary_gap}"

        logger.info(
            f"Generated follow-up query from primary knowledge gap: {follow_up_query}"
        )
        return [follow_up_query]

    def filter_valid_queries(
        self, queries: List[str], state: Dict[str, Any]
    ) -> List[str]:
        """Filter valid queries

        Args:
            queries: List of queries
            state: Current state

        Returns:
            Filtered list of valid queries
        """
        if not queries:
            return []

        # Get executed query history
        executed_queries = set()
        for record in self.research_history:
            if record.get("query"):
                executed_queries.add(record["query"].lower().strip())

        # Filter duplicate and invalid queries
        valid_queries = []
        for query in queries:
            if not query or not query.strip():
                continue

            query_normalized = query.lower().strip()

            # Check for duplicates
            if query_normalized in executed_queries:
                logger.debug(f"Skipping duplicate query: {query}")
                continue

            # Check query length
            if len(query.strip()) < 5:
                logger.debug(f"Skipping too short query: {query}")
                continue

            valid_queries.append(query)
            executed_queries.add(query_normalized)

        logger.info(
            f"Filtered {len(valid_queries)} valid queries from {len(queries)} total queries"
        )
        return valid_queries

    def _get_elapsed_time(self) -> float:
        """Get elapsed time (minutes)

        Returns:
            Elapsed time (minutes)
        """
        if not self.start_time:
            return 0.0

        elapsed = datetime.now() - self.start_time
        return elapsed.total_seconds() / 60.0

    def get_research_summary(self) -> Dict[str, Any]:
        """Get research summary

        Returns:
            Research summary dictionary
        """
        return {
            "iteration_count": self.iteration_count,
            "max_iterations": self.unified_config.max_follow_up_iterations,
            "elapsed_time_minutes": self._get_elapsed_time(),
            "max_time_minutes": getattr(
                self.unified_config, "max_research_time_minutes", 30
            ),
            "research_records_count": len(self.research_history),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "session_active": self.start_time is not None,
        }

    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get research history

        Returns:
            Research history list
        """
        # Return a copy of history records to avoid external modification
        return [record.copy() for record in self.research_history]

    async def execute_iterative_research(
        self,
        query: str,
        state: Dict[str, Any],
        reflection_result: Any = None,
        prompt_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute iterative research process

        Args:
            query: Research query
            state: Current state
            reflection_result: Reflection result (optional)
            prompt_context: Full prompt context with original_user_query, resources_info, etc. (optional)

        Returns:
            Research execution result
        """
        # --- Start of new code ---
        # Store the prompt context for use in research operations
        if prompt_context:
            self.prompt_context = prompt_context
        # --- End of new code ---
        logger.info(f"Starting iterative research execution for query: {query}")

        # Start research session if not already started
        if self.start_time is None:
            self.start_research_session()

        # Increment iteration
        current_iteration = self.increment_iteration()

        # Check termination conditions before execution
        should_terminate, reason, decision_factors = self.check_termination_conditions(
            state, reflection_result
        )

        if should_terminate:
            logger.info(f"Research terminated: {reason}")
            return {
                "terminated": True,
                "termination_reason": reason,
                "decision_factors": decision_factors,
                "iteration": current_iteration,
                "research_summary": self.get_research_summary(),
            }

        # Generate follow-up query if reflection result is available
        primary_follow_up_query = None
        if reflection_result:
            follow_up_queries = self.generate_follow_up_queries(
                reflection_result, state
            )
            follow_up_queries = self.filter_valid_queries(follow_up_queries, state)
            # Take the first valid query as the primary follow-up query
            primary_follow_up_query = follow_up_queries[0] if follow_up_queries else None

        # Execute research iteration
        execution_result = {
            "terminated": False,
            "iteration": current_iteration,
            "query": query,
            "primary_follow_up_query": primary_follow_up_query,
            "decision_factors": decision_factors,
            "research_summary": self.get_research_summary(),
        }

        # Add research record
        self.add_research_record(query, execution_result, reflection_result)

        logger.info(
            f"Iterative research execution completed for iteration {current_iteration}"
        )
        return execution_result
