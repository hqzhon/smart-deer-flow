# DeerFlow Collaboration Mechanism Optimization - Dynamic Role Assignment Mechanism

import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type enumeration"""
    DATA_ANALYSIS = "data_analysis"
    WEB_RESEARCH = "web_research" 
    CODE_GENERATION = "code_generation"
    CONTENT_WRITING = "content_writing"
    DOCUMENT_ANALYSIS = "document_analysis"
    FINANCIAL_ANALYSIS = "financial_analysis"
    TECHNICAL_RESEARCH = "technical_research"


@dataclass
class AgentCapability:
    """Agent capability assessment"""
    agent_id: str
    agent_type: str
    task_compatibility: Dict[TaskType, float] = field(default_factory=dict)
    historical_success_rate: float = 0.0
    tool_access_score: float = 0.0
    current_workload: int = 0
    specialized_domains: List[str] = field(default_factory=list)


@dataclass
class TaskRequirement:
    """Task requirement definition"""
    task_id: str
    task_type: TaskType
    description: str
    required_tools: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity_level: int = 1  # 1-5, 5 is most complex
    estimated_duration: int = 30  # Estimated minutes
    priority: int = 1  # 1-5, 5 is highest priority


@dataclass
class Bid:
    """Agent bid"""
    agent_id: str
    confidence_score: float
    reasoning: str
    estimated_time: int
    resource_requirements: List[str] = field(default_factory=list)


class RoleBiddingSystem:
    """Dynamic role assignment system"""
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self.task_history: List[Dict] = []
        self.active_tasks: Dict[str, str] = {}  # task_id -> agent_id
        
    def register_agent(self, agent: AgentCapability):
        """Register agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Register agent: {agent.agent_id} ({agent.agent_type})")
        
    def calculate_compatibility_score(self, agent: AgentCapability, task: TaskRequirement) -> float:
        """Calculate compatibility score between agent and task"""
        base_score = agent.task_compatibility.get(task.task_type, 0.1)
        
        # Historical success rate weight
        success_weight = agent.historical_success_rate * 0.3
        
        # Tool matching weight
        tool_match = 0.0
        if task.required_tools:
            # Simplified tool matching calculation
            tool_match = agent.tool_access_score * 0.2
            
        # Domain expertise weight
        domain_score = 0.0
        if task.domain and task.domain in agent.specialized_domains:
            domain_score = 0.2
            
        # Workload penalty
        workload_penalty = max(0, (agent.current_workload - 2) * 0.1)
        
        final_score = base_score + success_weight + tool_match + domain_score - workload_penalty
        return max(0.0, min(1.0, final_score))
    
    def generate_agent_bid(self, agent: AgentCapability, task: TaskRequirement) -> Optional[Bid]:
        """Generate agent bid"""
        compatibility = self.calculate_compatibility_score(agent, task)
        
        # Set bidding threshold
        if compatibility < 0.3:
            return None
            
        # Calculate confidence score
        confidence = compatibility * 0.7 + agent.historical_success_rate * 0.3
        
        # Generate bid reasoning
        reasoning = self._generate_bid_reasoning(agent, task, compatibility)
        
        # Estimate time (based on complexity and agent capability)
        estimated_time = int(task.estimated_duration * (1.5 - compatibility))
        
        return Bid(
            agent_id=agent.agent_id,
            confidence_score=confidence,
            reasoning=reasoning,
            estimated_time=estimated_time,
            resource_requirements=task.required_tools
        )
    
    def _generate_bid_reasoning(self, agent: AgentCapability, task: TaskRequirement, compatibility: float) -> str:
        """Generate bid reasoning"""
        reasons = []
        
        if compatibility > 0.7:
            reasons.append(f"High match for task type {task.task_type.value}")
        elif compatibility > 0.5:
            reasons.append(f"Moderate match for task type {task.task_type.value}")
            
        if agent.historical_success_rate > 0.8:
            reasons.append("Excellent historical success rate")
        elif agent.historical_success_rate > 0.6:
            reasons.append("Good historical success rate")
            
        if task.domain in agent.specialized_domains:
            reasons.append(f"Domain expertise match: {task.domain}")
            
        if agent.current_workload == 0:
            reasons.append("Currently idle, can execute immediately")
        elif agent.current_workload < 2:
            reasons.append("Light workload, sufficient time available")
            
        return "; ".join(reasons) if reasons else "Basic capability match"
    
    def conduct_bidding(self, task: TaskRequirement) -> List[Bid]:
        """Conduct bidding process"""
        logger.info(f"Start task bidding: {task.task_id} ({task.task_type.value})")
        
        bids = []
        for agent in self.agents.values():
            bid = self.generate_agent_bid(agent, task)
            if bid:
                bids.append(bid)
                logger.debug(f"Agent {agent.agent_id} bid, confidence: {bid.confidence_score:.2f}")
        
        # Sort by confidence
        bids.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Received {len(bids)} valid bids")
        return bids
    
    def select_winner(self, task: TaskRequirement, bids: List[Bid]) -> Optional[str]:
        """Select winning agent"""
        if not bids:
            logger.warning(f"Task {task.task_id} received no bids")
            return None
            
        # Comprehensive scoring: confidence + time efficiency + randomness
        scored_bids = []
        for bid in bids[:3]:  # Only consider top 3
            time_efficiency = 1.0 - (bid.estimated_time / (task.estimated_duration * 2))
            time_efficiency = max(0.0, time_efficiency)
            
            final_score = (
                bid.confidence_score * 0.6 +
                time_efficiency * 0.3 +
                0.1  # Base score to avoid complete determinism
            )
            
            scored_bids.append((bid.agent_id, final_score, bid))
            
        # Select highest score
        winner_id, best_score, winning_bid = max(scored_bids, key=lambda x: x[1])
        
        # Update agent workload
        if winner_id in self.agents:
            self.agents[winner_id].current_workload += 1
            
        # Record assignment result
        self.active_tasks[task.task_id] = winner_id
        
        logger.info(f"Task {task.task_id} assigned to agent {winner_id}, score: {best_score:.2f}")
        logger.debug(f"Winning reason: {winning_bid.reasoning}")
        
        return winner_id
    
    def assign_task(self, task: TaskRequirement) -> Optional[str]:
        """Assign task (complete process)"""
        bids = self.conduct_bidding(task)
        return self.select_winner(task, bids)
    
    def complete_task(self, task_id: str, success: bool, actual_duration: int):
        """Update after task completion"""
        if task_id not in self.active_tasks:
            logger.warning(f"Active task not found: {task_id}")
            return
            
        agent_id = self.active_tasks[task_id]
        
        # Update agent workload
        if agent_id in self.agents:
            self.agents[agent_id].current_workload = max(0, 
                self.agents[agent_id].current_workload - 1)
            
            # Update historical success rate
            current_rate = self.agents[agent_id].historical_success_rate
            # Use exponential moving average for update
            new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            self.agents[agent_id].historical_success_rate = new_rate
            
        # Record to history
        self.task_history.append({
            "task_id": task_id,
            "agent_id": agent_id,
            "success": success,
            "duration": actual_duration,
            "timestamp": logger.name  # Simplified timestamp
        })
        
        # Remove active task
        del self.active_tasks[task_id]
        
        logger.info(f"Task {task_id} completed, success: {success}, duration: {actual_duration} minutes")
    
    def get_agent_statistics(self) -> Dict[str, Dict]:
        """Get agent statistics"""
        stats = {}
        for agent_id, agent in self.agents.items():
            completed_tasks = [t for t in self.task_history if t["agent_id"] == agent_id]
            success_count = sum(1 for t in completed_tasks if t["success"])
            
            stats[agent_id] = {
                "agent_type": agent.agent_type,
                "historical_success_rate": agent.historical_success_rate,
                "completed_tasks": len(completed_tasks),
                "success_count": success_count,
                "current_workload": agent.current_workload,
                "specialized_domains": agent.specialized_domains
            }
            
        return stats


# Predefined agent configurations
def create_default_agents() -> List[AgentCapability]:
    """Create default agent configurations"""
    agents = [
        AgentCapability(
            agent_id="researcher_01",
            agent_type="researcher", 
            task_compatibility={
                TaskType.WEB_RESEARCH: 0.9,
                TaskType.DOCUMENT_ANALYSIS: 0.8,
                TaskType.TECHNICAL_RESEARCH: 0.7,
                TaskType.CONTENT_WRITING: 0.6
            },
            historical_success_rate=0.85,
            tool_access_score=0.8,
            specialized_domains=["academic", "technology", "science"]
        ),
        AgentCapability(
            agent_id="data_analyst_01",
            agent_type="coder",
            task_compatibility={
                TaskType.DATA_ANALYSIS: 0.95,
                TaskType.CODE_GENERATION: 0.8,
                TaskType.FINANCIAL_ANALYSIS: 0.9,
                TaskType.WEB_RESEARCH: 0.6
            },
            historical_success_rate=0.90,
            tool_access_score=0.9,
            specialized_domains=["data_science", "finance", "statistics"]
        ),
        AgentCapability(
            agent_id="content_writer_01",
            agent_type="reporter",
            task_compatibility={
                TaskType.CONTENT_WRITING: 0.95,
                TaskType.DOCUMENT_ANALYSIS: 0.7,
                TaskType.WEB_RESEARCH: 0.6,
                TaskType.TECHNICAL_RESEARCH: 0.5
            },
            historical_success_rate=0.88,
            tool_access_score=0.7,
            specialized_domains=["journalism", "marketing", "communication"]
        ),
        AgentCapability(
            agent_id="code_specialist_01",
            agent_type="coder",
            task_compatibility={
                TaskType.CODE_GENERATION: 0.95,
                TaskType.DATA_ANALYSIS: 0.8,
                TaskType.TECHNICAL_RESEARCH: 0.7,
                TaskType.DOCUMENT_ANALYSIS: 0.6
            },
            historical_success_rate=0.92,
            tool_access_score=0.95,
            specialized_domains=["programming", "software_engineering", "automation"]
        )
    ]
    
    return agents


# Usage example
if __name__ == "__main__":
    # Create bidding system
    bidding_system = RoleBiddingSystem()
    
    # Register agents
    for agent in create_default_agents():
        bidding_system.register_agent(agent)
    
    # Create example task
    task = TaskRequirement(
        task_id="task_001",
        task_type=TaskType.DATA_ANALYSIS,
        description="Analyze sales data trends",
        required_tools=["python", "pandas", "matplotlib"],
        domain="finance",
        complexity_level=3,
        estimated_duration=45,
        priority=4
    )
    
    # Assign task
    assigned_agent = bidding_system.assign_task(task)
    if assigned_agent:
        print(f"Task assigned to: {assigned_agent}")
        
        # Simulate task completion
        bidding_system.complete_task("task_001", success=True, actual_duration=40)
        
        # View statistics
        stats = bidding_system.get_agent_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
