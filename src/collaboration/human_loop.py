# DeerFlow Collaboration Mechanism Optimization - Real-time Human Feedback Intervention Mechanism (Human-in-the-loop 2.0)

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Intervention types"""
    TASK_MODIFICATION = "task_modification"  # Task modification
    PARAMETER_ADJUSTMENT = "parameter_adjustment"  # Parameter adjustment
    TOOL_SELECTION = "tool_selection"  # Tool selection
    OUTPUT_REFINEMENT = "output_refinement"  # Output refinement
    ALTERNATIVE_APPROACH = "alternative_approach"  # Alternative approach
    VALIDATION_REQUEST = "validation_request"  # Validation request


class InterventionPriority(Enum):
    """Intervention priority"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class InterventionPoint:
    """Intervention point definition"""
    point_id: str
    task_id: str
    agent_id: str
    intervention_type: InterventionType
    description: str
    current_state: Dict[str, Any]
    suggested_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    priority: InterventionPriority = InterventionPriority.MEDIUM
    auto_timeout: int = 300  # Auto timeout (seconds)
    requires_human_decision: bool = True


@dataclass
class HumanFeedback:
    """Human feedback"""
    feedback_id: str
    intervention_point_id: str
    decision: str
    reasoning: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskState:
    """Task state snapshot"""
    task_id: str
    agent_id: str
    step_name: str
    input_data: Dict[str, Any]
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    current_tools: List[str] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryModule:
    """Memory module - Reference DAgent's memory system"""
    
    def __init__(self):
        self.task_memories: Dict[str, List[TaskState]] = {}
        self.successful_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        
    def record_task_state(self, state: TaskState):
        """Record task state"""
        if state.task_id not in self.task_memories:
            self.task_memories[state.task_id] = []
        self.task_memories[state.task_id].append(state)
        logger.debug(f"Record task state: {state.task_id} - {state.step_name}")
        
    def get_task_history(self, task_id: str) -> List[TaskState]:
        """Get task history"""
        return self.task_memories.get(task_id, [])
        
    def find_similar_patterns(self, current_state: TaskState) -> List[Dict[str, Any]]:
        """Find similar patterns"""
        similar_patterns = []
        
        # Simplified pattern matching logic
        for pattern in self.successful_patterns:
            if (pattern.get("agent_type") == current_state.agent_id and
                pattern.get("task_type") in current_state.metadata.get("task_type", "")):
                similar_patterns.append(pattern)
                
        return similar_patterns[:3]  # Return the 3 most relevant patterns
        
    def record_success_pattern(self, task_id: str, pattern: Dict[str, Any]):
        """Record success pattern"""
        pattern["task_id"] = task_id
        pattern["timestamp"] = datetime.now().isoformat()
        self.successful_patterns.append(pattern)
        logger.info(f"Record success pattern: {task_id}")
        
    def record_failure_pattern(self, task_id: str, pattern: Dict[str, Any]):
        """Record failure pattern"""
        pattern["task_id"] = task_id
        pattern["timestamp"] = datetime.now().isoformat()
        self.failure_patterns.append(pattern)
        logger.info(f"Record failure pattern: {task_id}")


class ReflectionValidator:
    """Reflection validator - Reference AMiner's reflection-validation loop"""
    
    def __init__(self):
        self.validation_rules: List[Callable] = []
        self.confidence_threshold = 0.7
        
    def add_validation_rule(self, rule: Callable[[Dict[str, Any]], float]):
        """Add validation rule"""
        self.validation_rules.append(rule)
        
    def validate_intermediate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate intermediate result"""
        validation_scores = []
        issues = []
        
        for rule in self.validation_rules:
            try:
                score = rule(result)
                validation_scores.append(score)
            except Exception as e:
                logger.warning(f"Validation rule execution failed: {e}")
                issues.append(f"Validation rule exception: {str(e)}")
                
        avg_confidence = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        
        return {
            "confidence": avg_confidence,
            "requires_human_review": avg_confidence < self.confidence_threshold,
            "issues": issues,
            "detailed_scores": validation_scores
        }
        
    def generate_alternatives(self, current_approach: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternatives"""
        alternatives = []
        
        # Generate alternatives based on current method
        if current_approach.get("method") == "sql_query":
            alternatives.append({
                "method": "pandas_analysis",
                "description": "Use Pandas for data analysis",
                "confidence": 0.8,
                "estimated_time": current_approach.get("estimated_time", 30) * 1.2
            })
            alternatives.append({
                "method": "statistical_analysis",
                "description": "Use statistical analysis methods",
                "confidence": 0.7,
                "estimated_time": current_approach.get("estimated_time", 30) * 1.5
            })
            
        return alternatives


class HumanLoopController:
    """Human-loop controller - Core coordination component"""
    
    def __init__(self):
        self.memory = MemoryModule()
        self.validator = ReflectionValidator()
        self.intervention_points: Dict[str, InterventionPoint] = {}
        self.active_tasks: Dict[str, TaskState] = {}
        self.feedback_queue: List[HumanFeedback] = []
        
        # Set up default validation rules
        self._setup_default_validation_rules()
        
    def _setup_default_validation_rules(self):
        """Set up default validation rules"""
        
        def confidence_rule(result: Dict[str, Any]) -> float:
            """Confidence-based validation rule"""
            return result.get("confidence", 0.5)
            
        def consistency_rule(result: Dict[str, Any]) -> float:
            """Consistency validation rule"""
            # Simplified consistency check
            if "data" in result and "conclusion" in result:
                return 0.8  # Assume data and conclusion are consistent
            return 0.3
            
        self.validator.add_validation_rule(confidence_rule)
        self.validator.add_validation_rule(consistency_rule)
        
    async def monitor_task_execution(self, task_id: str, agent_id: str, execution_steps: List[Dict[str, Any]]):
        """Monitor task execution process"""
        logger.info(f"Start monitoring task execution: {task_id}")
        
        for step_index, step in enumerate(execution_steps):
            # Record current state
            current_state = TaskState(
                task_id=task_id,
                agent_id=agent_id,
                step_name=step.get("name", f"step_{step_index}"),
                input_data=step.get("input", {}),
                current_tools=step.get("tools", []),
                execution_path=[s.get("name", f"step_{i}") for i, s in enumerate(execution_steps[:step_index+1])],
                metadata=step.get("metadata", {})
            )
            
            self.memory.record_task_state(current_state)
            
            # Check if human intervention is needed
            intervention_needed = await self._check_intervention_triggers(current_state, step)
            
            if intervention_needed:
                intervention_point = await self._create_intervention_point(current_state, step)
                await self._handle_intervention(intervention_point)
                
    async def _check_intervention_triggers(self, state: TaskState, step: Dict[str, Any]) -> bool:
        """Check intervention trigger conditions"""
        triggers = []
        
        # 1. Complex task auto trigger
        if step.get("complexity", 1) > 3:
            triggers.append("High complexity task")
            
        # 2. Multi-tool selection trigger
        if len(state.current_tools) > 3:
            triggers.append("Multi-tool selection scenario")
            
        # 3. Historical failure pattern trigger
        similar_failures = [p for p in self.memory.failure_patterns 
                          if p.get("agent_type") == state.agent_id]
        if len(similar_failures) > 2:
            triggers.append("Historical failure pattern match")
            
        # 4. Intermediate result validation failure
        if step.get("intermediate_result"):
            validation = self.validator.validate_intermediate_result(
                step["intermediate_result"]
            )
            if validation["requires_human_review"]:
                triggers.append("Intermediate result validation failure")
                
        if triggers:
            logger.info(f"Trigger human intervention: {', '.join(triggers)}")
            return True
            
        return False
        
    async def _create_intervention_point(self, state: TaskState, step: Dict[str, Any]) -> InterventionPoint:
        """创建干预点"""
        point_id = str(uuid.uuid4())
        
        # 生成替代方案
        alternatives = self.reflection_validator.generate_alternatives(step)
        
        # 确定干预类型
        intervention_type = InterventionType.VALIDATION_REQUEST
        if len(alternatives) > 1:
            intervention_type = InterventionType.ALTERNATIVE_APPROACH
        elif step.get("parameters"):
            intervention_type = InterventionType.PARAMETER_ADJUSTMENT
            
        intervention_point = InterventionPoint(
            point_id=point_id,
            task_id=state.task_id,
            agent_id=state.agent_id,
            intervention_type=intervention_type,
            description=f"步骤 '{state.step_name}' 需要人类决策",
            current_state={
                "step_name": state.step_name,
                "input_data": state.input_data,
                "tools": state.current_tools,
                "execution_path": state.execution_path
            },
            suggested_alternatives=alternatives,
            priority=InterventionPriority.MEDIUM
        )
        
        self.intervention_points[point_id] = intervention_point
        return intervention_point
        
    async def _handle_intervention(self, intervention_point: InterventionPoint):
        """处理干预"""
        logger.info(f"处理干预点: {intervention_point.point_id}")
        
        if self.auto_decision_enabled and intervention_point.priority.value <= 2:
            # 低优先级自动决策
            decision = await self._make_auto_decision(intervention_point)
            await self._apply_decision(intervention_point.point_id, decision)
        else:
            # 等待人类决策
            await self._request_human_decision(intervention_point)
            
    async def _make_auto_decision(self, intervention_point: InterventionPoint) -> Dict[str, Any]:
        """自动决策"""
        logger.info(f"执行自动决策: {intervention_point.point_id}")
        
        # Make decisions based on historical success patterns
        similar_patterns = self.memory.find_similar_patterns(
            TaskState(
                task_id=intervention_point.task_id,
                agent_id=intervention_point.agent_id,
                step_name="auto_decision",
                input_data=intervention_point.current_state
            )
        )
        
        if similar_patterns:
            # 选择成功率最高的模式
            best_pattern = max(similar_patterns, key=lambda x: x.get("success_rate", 0))
            return {
                "decision": "apply_pattern",
                "pattern": best_pattern,
                "reasoning": f"Based on historical success pattern, success rate: {best_pattern.get('success_rate', 0):.2f}"
            }
        
        # 选择第一个替代方案
        if intervention_point.suggested_alternatives:
            best_alternative = max(intervention_point.suggested_alternatives, 
                                 key=lambda x: x.get("confidence", 0))
            return {
                "decision": "use_alternative",
                "alternative": best_alternative,
                "reasoning": f"Choose alternative with highest confidence: {best_alternative.get('confidence', 0):.2f}"
            }
            
        return {
            "decision": "continue",
            "reasoning": "No better alternatives, continue current execution"
        }
        
    async def _request_human_decision(self, intervention_point: InterventionPoint):
        """请求人类决策"""
        logger.info(f"请求人类决策: {intervention_point.point_id}")
        
        # Generate decision options
        options = {
            "continue": "Continue current execution path",
            "modify": "Modify current parameters or method",
            "alternative": "Choose alternative approach",
            "abort": "Abort current step"
        }
        
        if intervention_point.suggested_alternatives:
            for i, alt in enumerate(intervention_point.suggested_alternatives):
                options[f"alt_{i}"] = f"Alternative {i+1}: {alt.get('description', 'Unknown')}"
                
        request_data = {
            "intervention_id": intervention_point.point_id,
            "task_id": intervention_point.task_id,
            "description": intervention_point.description,
            "current_state": intervention_point.current_state,
            "options": options,
            "suggested_alternatives": intervention_point.suggested_alternatives,
            "priority": intervention_point.priority.name,
            "auto_timeout": intervention_point.auto_timeout
        }
        
        # In actual implementation, this would be sent to frontend via WebSocket or other mechanism
        logger.info(f"Human decision request: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        # Simulate waiting for human response (in actual implementation this would be real async waiting)
        await asyncio.sleep(1)
        
        # Simulate human feedback
        mock_feedback = HumanFeedback(
            feedback_id=str(uuid.uuid4()),
            intervention_point_id=intervention_point.point_id,
            decision="continue",
            reasoning="Current method looks reasonable, continue execution",
            modifications={}
        )
        
        await self._apply_human_feedback(mock_feedback)
        
    async def _apply_human_feedback(self, feedback: HumanFeedback):
        """应用人类反馈"""
        self.feedback_queue.append(feedback)
        await self._apply_decision(feedback.intervention_point_id, {
            "decision": feedback.decision,
            "reasoning": feedback.reasoning,
            "modifications": feedback.modifications
        })
        
    async def _apply_decision(self, intervention_point_id: str, decision: Dict[str, Any]):
        """应用决策"""
        if intervention_point_id not in self.intervention_points:
            logger.warning(f"Intervention point not found: {intervention_point_id}")
            return
            
        intervention_point = self.intervention_points[intervention_point_id]
        logger.info(f"Apply decision: {decision['decision']} - {decision['reasoning']}")
        
        # Execute corresponding operations based on decision type
        if decision["decision"] == "apply_pattern":
            await self._apply_pattern(intervention_point, decision["pattern"])
        elif decision["decision"] == "use_alternative":
            await self._apply_alternative(intervention_point, decision["alternative"])
        elif decision["decision"] == "modify":
            await self._apply_modifications(intervention_point, decision.get("modifications", {}))
            
        # Remove processed intervention point
        del self.intervention_points[intervention_point_id]
        
    async def _apply_pattern(self, intervention_point: InterventionPoint, pattern: Dict[str, Any]):
        """Apply historical success pattern"""
        logger.info(f"Apply historical pattern: {pattern.get('description', 'Unknown pattern')}")
        # In actual implementation, execution plan would be modified
        
    async def _apply_alternative(self, intervention_point: InterventionPoint, alternative: Dict[str, Any]):
        """Apply alternative approach"""
        logger.info(f"Apply alternative approach: {alternative.get('description', 'Unknown approach')}")
        # In actual implementation, would switch to alternative approach
        
    async def _apply_modifications(self, intervention_point: InterventionPoint, modifications: Dict[str, Any]):
        """Apply modifications"""
        logger.info(f"Apply modifications: {modifications}")
        # In actual implementation, parameter modifications would be applied
        
    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get intervention summary"""
        return {
            "active_interventions": len(self.intervention_points),
            "total_feedback": len(self.feedback_queue),
            "auto_decisions": sum(1 for f in self.feedback_queue if f.decision.startswith("auto_")),
            "human_decisions": sum(1 for f in self.feedback_queue if not f.decision.startswith("auto_")),
            "recent_patterns": len(self.memory.successful_patterns[-10:])
        }


# Usage example
async def demo_human_loop():
    """Demonstrate human intervention system"""
    controller = HumanLoopController()
    
    # Simulate task execution steps
    execution_steps = [
        {
            "name": "data_collection",
            "input": {"source": "database", "query": "SELECT * FROM sales"},
            "tools": ["sql_tool", "pandas"],
            "complexity": 2,
            "metadata": {"task_type": "data_analysis"}
        },
        {
            "name": "data_analysis", 
            "input": {"data": "sales_data.csv"},
            "tools": ["pandas", "matplotlib", "seaborn", "numpy"],
            "complexity": 4,
            "intermediate_result": {
                "status": "partial",
                "content": "Data analysis in progress",
                "data": {"rows": 1000, "columns": 15}
            },
            "metadata": {"task_type": "data_analysis"}
        },
        {
            "name": "visualization",
            "input": {"analysis_results": "trend_data"},
            "tools": ["matplotlib"],
            "complexity": 3,
            "metadata": {"task_type": "visualization"}
        }
    ]
    
    # Monitor task execution
    await controller.monitor_task_execution("demo_task_001", "data_analyst_01", execution_steps)
    
    # Display summary
    summary = controller.get_intervention_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(demo_human_loop())
