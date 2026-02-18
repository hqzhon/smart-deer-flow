import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)

_PLANNING_TOOL_DESCRIPTION = """
A planning tool that allows the agent to create and manage plans for solving complex tasks.
The tool provides functionality for creating plans, updating plan steps, and tracking progress.

Commands:
- create: Create a new plan with title and steps
- update: Update an existing plan
- list: List all available plans
- get: Get details of a specific plan
- set_active: Set a plan as the active plan
- mark_step: Mark a step with a specific status
- delete: Delete a plan
"""


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class Plan(BaseModel):
    """Plan model for storing plan data"""

    plan_id: str
    title: str
    steps: List[str] = Field(default_factory=list)
    step_statuses: List[str] = Field(default_factory=list)
    step_notes: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanningTool(BaseTool):
    """
    A planning tool that allows the agent to create and manage plans for solving complex tasks.
    The tool provides functionality for creating plans, updating plan steps, and tracking progress.
    """

    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to execute. Available commands: create, update, list, get, set_active, mark_step, delete.",
                    "enum": [
                        "create",
                        "update",
                        "list",
                        "get",
                        "set_active",
                        "mark_step",
                        "delete",
                    ],
                    "type": "string",
                },
                "plan_id": {
                    "description": "Unique identifier for the plan. Required for create, update, set_active, and delete commands.",
                    "type": "string",
                },
                "title": {
                    "description": "Title for the plan. Required for create command, optional for update command.",
                    "type": "string",
                },
                "steps": {
                    "description": "List of plan steps. Required for create command, optional for update command.",
                    "type": "array",
                    "items": {"type": "string"},
                },
                "step_index": {
                    "description": "Index of the step to update (0-based). Required for mark_step command.",
                    "type": "integer",
                },
                "step_status": {
                    "description": "Status to set for a step. Used with mark_step command.",
                    "enum": ["not_started", "in_progress", "completed", "blocked"],
                    "type": "string",
                },
                "step_notes": {
                    "description": "Additional notes for a step. Optional for mark_step command.",
                    "type": "string",
                },
            },
            "required": ["command"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["command"]

    @property
    def category(self) -> str:
        return "planning"

    @property
    def tags(self) -> list[str]:
        return ["planning", "workflow", "task-management"]

    def __init__(self, **data):
        super().__init__(**data)
        self._plans: Dict[str, Plan] = {}
        self._current_plan_id: Optional[str] = None

    def execute(self, **kwargs) -> ToolResult:
        command = kwargs.get("command")
        if not command:
            return ToolResult(error="Parameter 'command' is required")

        if command == "create":
            return self._create_plan(
                kwargs.get("plan_id"),
                kwargs.get("title"),
                kwargs.get("steps"),
                kwargs.get("metadata"),
            )
        elif command == "update":
            return self._update_plan(
                kwargs.get("plan_id"),
                kwargs.get("title"),
                kwargs.get("steps"),
            )
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(kwargs.get("plan_id"))
        elif command == "set_active":
            return self._set_active_plan(kwargs.get("plan_id"))
        elif command == "mark_step":
            return self._mark_step(
                kwargs.get("plan_id"),
                kwargs.get("step_index"),
                kwargs.get("step_status"),
                kwargs.get("step_notes"),
            )
        elif command == "delete":
            return self._delete_plan(kwargs.get("plan_id"))
        else:
            return ToolResult(
                error=f"Unrecognized command: {command}. Allowed commands are: create, update, list, get, set_active, mark_step, delete"
            )

    def _create_plan(
        self,
        plan_id: Optional[str],
        title: Optional[str],
        steps: Optional[List[str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if not plan_id:
            return ToolResult(
                error="Parameter 'plan_id' is required for command: create"
            )

        if plan_id in self._plans:
            return ToolResult(
                error=f"A plan with ID '{plan_id}' already exists. Use 'update' to modify existing plans."
            )

        if not title:
            return ToolResult(error="Parameter 'title' is required for command: create")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            return ToolResult(
                error="Parameter 'steps' must be a non-empty list of strings for command: create"
            )

        plan = Plan(
            plan_id=plan_id,
            title=title,
            steps=steps,
            step_statuses=[PlanStepStatus.NOT_STARTED.value] * len(steps),
            step_notes=[""] * len(steps),
            metadata=metadata or {},
        )

        self._plans[plan_id] = plan
        self._current_plan_id = plan_id

        return ToolResult(
            output=f"Plan created successfully with ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self,
        plan_id: Optional[str],
        title: Optional[str],
        steps: Optional[List[str]],
    ) -> ToolResult:
        if not plan_id:
            return ToolResult(
                error="Parameter 'plan_id' is required for command: update"
            )

        if plan_id not in self._plans:
            return ToolResult(error=f"No plan found with ID: {plan_id}")

        plan = self._plans[plan_id]

        if title:
            plan.title = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                return ToolResult(
                    error="Parameter 'steps' must be a list of strings for command: update"
                )

            old_steps = plan.steps
            old_statuses = plan.step_statuses
            old_notes = plan.step_notes

            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append(PlanStepStatus.NOT_STARTED.value)
                    new_notes.append("")

            plan.steps = steps
            plan.step_statuses = new_statuses
            plan.step_notes = new_notes

        plan.updated_at = time.time()

        return ToolResult(
            output=f"Plan updated successfully: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        if not self._plans:
            return ToolResult(
                output="No plans available. Create a plan with the 'create' command."
            )

        output = "Available plans:\n"
        for plan_id, plan in self._plans.items():
            current_marker = " (active)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1
                for status in plan.step_statuses
                if status == PlanStepStatus.COMPLETED.value
            )
            total = len(plan.steps)
            progress = f"{completed}/{total} steps completed"
            output += f"• {plan_id}{current_marker}: {plan.title} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        if not plan_id:
            if not self._current_plan_id:
                return ToolResult(
                    error="No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self._plans:
            return ToolResult(error=f"No plan found with ID: {plan_id}")

        plan = self._plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        if not plan_id:
            return ToolResult(
                error="Parameter 'plan_id' is required for command: set_active"
            )

        if plan_id not in self._plans:
            return ToolResult(error=f"No plan found with ID: {plan_id}")

        self._current_plan_id = plan_id
        plan = self._plans[plan_id]
        return ToolResult(
            output=f"Plan '{plan_id}' is now the active plan.\n\n{self._format_plan(plan)}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        if not plan_id:
            if not self._current_plan_id:
                return ToolResult(
                    error="No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self._plans:
            return ToolResult(error=f"No plan found with ID: {plan_id}")

        if step_index is None:
            return ToolResult(
                error="Parameter 'step_index' is required for command: mark_step"
            )

        plan = self._plans[plan_id]

        if step_index < 0 or step_index >= len(plan.steps):
            return ToolResult(
                error=f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(plan.steps) - 1}."
            )

        if step_status and step_status not in PlanStepStatus.get_all_statuses():
            return ToolResult(
                error=f"Invalid step_status: {step_status}. Valid statuses are: {', '.join(PlanStepStatus.get_all_statuses())}"
            )

        if step_status:
            plan.step_statuses[step_index] = step_status

        if step_notes:
            plan.step_notes[step_index] = step_notes

        plan.updated_at = time.time()

        return ToolResult(
            output=f"Step {step_index} updated in plan '{plan_id}'.\n\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        if not plan_id:
            return ToolResult(
                error="Parameter 'plan_id' is required for command: delete"
            )

        if plan_id not in self._plans:
            return ToolResult(error=f"No plan found with ID: {plan_id}")

        del self._plans[plan_id]

        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"Plan '{plan_id}' has been deleted.")

    def _format_plan(self, plan: Plan) -> str:
        output = f"Plan: {plan.title} (ID: {plan.plan_id})\n"
        output += "=" * min(len(output), 60) + "\n\n"

        total_steps = len(plan.steps)
        completed = sum(
            1
            for status in plan.step_statuses
            if status == PlanStepStatus.COMPLETED.value
        )
        in_progress = sum(
            1
            for status in plan.step_statuses
            if status == PlanStepStatus.IN_PROGRESS.value
        )
        blocked = sum(
            1 for status in plan.step_statuses if status == PlanStepStatus.BLOCKED.value
        )
        not_started = sum(
            1
            for status in plan.step_statuses
            if status == PlanStepStatus.NOT_STARTED.value
        )

        output += f"Progress: {completed}/{total_steps} steps completed "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {completed} completed, {in_progress} in progress, {blocked} blocked, {not_started} not started\n\n"
        output += "Steps:\n"

        status_marks = PlanStepStatus.get_status_marks()

        for i, (step, status, notes) in enumerate(
            zip(plan.steps, plan.step_statuses, plan.step_notes)
        ):
            status_symbol = status_marks.get(
                status, status_marks[PlanStepStatus.NOT_STARTED.value]
            )
            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   Notes: {notes}\n"

        return output

    def get_active_plan(self) -> Optional[Plan]:
        if self._current_plan_id and self._current_plan_id in self._plans:
            return self._plans[self._current_plan_id]
        return None

    def get_next_step(self) -> Optional[tuple[int, str]]:
        plan = self.get_active_plan()
        if not plan:
            return None

        for i, status in enumerate(plan.step_statuses):
            if status in PlanStepStatus.get_active_statuses():
                return i, plan.steps[i]
        return None

    def get_progress(self) -> Dict[str, Any]:
        plan = self.get_active_plan()
        if not plan:
            return {"error": "No active plan"}

        total = len(plan.steps)
        completed = sum(
            1 for s in plan.step_statuses if s == PlanStepStatus.COMPLETED.value
        )
        in_progress = sum(
            1 for s in plan.step_statuses if s == PlanStepStatus.IN_PROGRESS.value
        )
        blocked = sum(
            1 for s in plan.step_statuses if s == PlanStepStatus.BLOCKED.value
        )

        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "total_steps": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "not_started": total - completed - in_progress - blocked,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
        }
