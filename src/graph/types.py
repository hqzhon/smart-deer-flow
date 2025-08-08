# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import MessagesState
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from src.models.planner_model import Plan
from src.rag import Resource


class ReflectionState(BaseModel):
    """Independent reflection state model for better encapsulation and maintainability."""

    # Basic reflection control
    enabled: bool = Field(default=False, description="Whether reflection is enabled")
    count: int = Field(default=0, description="Number of reflection executions")
    integration_active: bool = Field(
        default=False, description="Whether reflection integration is active"
    )

    # Session management
    current_session: Optional[str] = Field(
        default=None, description="Current reflection session ID"
    )
    triggered: bool = Field(
        default=False, description="Whether reflection has been triggered"
    )
    last_step: int = Field(
        default=0, description="Last step where reflection was executed"
    )

    # Reflection results and analysis
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Historical reflection results"
    )
    comprehensive_report: str = Field(
        default="",
        description="Comprehensive research report synthesizing all findings",
    )
    primary_knowledge_gap: str = Field(
        default="", description="Primary identified knowledge gap"
    )
    primary_follow_up_query: str = Field(
        default="", description="Primary follow-up query"
    )

    # Assessment metrics
    sufficiency_score: float = Field(
        default=0.0, description="Research sufficiency score"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            # Custom encoders if needed for specific types
        },
    )


class State(MessagesState):
    """State for the agent system, extends MessagesState with modular reflection state."""

    # Runtime Variables
    locale: str = "en-US"
    research_topic: str = ""
    observations: list[str] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    final_report: str = ""
    auto_accepted_plan: bool = False
    enable_background_investigation: bool = True
    background_investigation_results: str = None
    enable_collaboration: bool = True
    collaboration_systems: dict = None
    agent_configurable: Optional[Any] = (
        None  # Store configurable object for research components
    )

    # Modular Reflection State
    reflection: ReflectionState = Field(default_factory=ReflectionState)

    # Backward compatibility properties for existing code
    @property
    def reflection_enabled(self) -> bool:
        """Backward compatibility: access reflection.enabled"""
        return self.reflection.enabled

    @reflection_enabled.setter
    def reflection_enabled(self, value: bool):
        """Backward compatibility: set reflection.enabled"""
        self.reflection.enabled = value

    @property
    def reflection_count(self) -> int:
        """Backward compatibility: access reflection.count"""
        return self.reflection.count

    @reflection_count.setter
    def reflection_count(self, value: int):
        """Backward compatibility: set reflection.count"""
        self.reflection.count = value

    @property
    def reflection_results(self) -> List[Dict[str, Any]]:
        """Backward compatibility: access reflection.results"""
        return self.reflection.results

    @reflection_results.setter
    def reflection_results(self, value: List[Dict[str, Any]]):
        """Backward compatibility: set reflection.results"""
        self.reflection.results = value

    @property
    def current_reflection_session(self) -> Optional[str]:
        """Backward compatibility: access reflection.current_session"""
        return self.reflection.current_session

    @current_reflection_session.setter
    def current_reflection_session(self, value: Optional[str]):
        """Backward compatibility: set reflection.current_session"""
        self.reflection.current_session = value

    @property
    def knowledge_gaps(self) -> List[str]:
        """Backward compatibility: access reflection.primary_knowledge_gap as list"""
        return (
            [self.reflection.primary_knowledge_gap]
            if self.reflection.primary_knowledge_gap
            else []
        )

    @knowledge_gaps.setter
    def knowledge_gaps(self, value: List[str]):
        """Backward compatibility: set reflection.primary_knowledge_gap from list"""
        self.reflection.primary_knowledge_gap = value[0] if value else ""

    @property
    def follow_up_queries(self) -> List[str]:
        """Backward compatibility: access reflection.primary_follow_up_query as list"""
        return (
            [self.reflection.primary_follow_up_query]
            if self.reflection.primary_follow_up_query
            else []
        )

    @follow_up_queries.setter
    def follow_up_queries(self, value: List[str]):
        """Backward compatibility: set reflection.primary_follow_up_query from list"""
        self.reflection.primary_follow_up_query = value[0] if value else ""

    @property
    def research_sufficiency_score(self) -> float:
        """Backward compatibility: access reflection.sufficiency_score"""
        return self.reflection.sufficiency_score

    @research_sufficiency_score.setter
    def research_sufficiency_score(self, value: float):
        """Backward compatibility: set reflection.sufficiency_score"""
        self.reflection.sufficiency_score = value

    @property
    def last_reflection_step(self) -> int:
        """Backward compatibility: access reflection.last_step"""
        return self.reflection.last_step

    @last_reflection_step.setter
    def last_reflection_step(self, value: int):
        """Backward compatibility: set reflection.last_step"""
        self.reflection.last_step = value

    @property
    def reflection_triggered(self) -> bool:
        """Backward compatibility: access reflection.triggered"""
        return self.reflection.triggered

    @reflection_triggered.setter
    def reflection_triggered(self, value: bool):
        """Backward compatibility: set reflection.triggered"""
        self.reflection.triggered = value

    @property
    def reflection_integration_active(self) -> bool:
        """Backward compatibility: access reflection.integration_active"""
        return self.reflection.integration_active

    @reflection_integration_active.setter
    def reflection_integration_active(self, value: bool):
        """Backward compatibility: set reflection.integration_active"""
        self.reflection.integration_active = value

    @property
    def comprehensive_report(self) -> str:
        """Backward compatibility: access reflection.comprehensive_report"""
        return self.reflection.comprehensive_report

    @comprehensive_report.setter
    def comprehensive_report(self, value: str):
        """Backward compatibility: set reflection.comprehensive_report"""
        self.reflection.comprehensive_report = value
