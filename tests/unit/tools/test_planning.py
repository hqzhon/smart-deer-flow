from src.tools.planning.planning_tool import (
    PlanningTool,
    Plan,
    PlanStepStatus,
)


class TestPlanStepStatus:
    def test_get_all_statuses(self):
        statuses = PlanStepStatus.get_all_statuses()
        assert "not_started" in statuses
        assert "in_progress" in statuses
        assert "completed" in statuses
        assert "blocked" in statuses

    def test_get_active_statuses(self):
        active = PlanStepStatus.get_active_statuses()
        assert "not_started" in active
        assert "in_progress" in active
        assert "completed" not in active
        assert "blocked" not in active

    def test_get_status_marks(self):
        marks = PlanStepStatus.get_status_marks()
        assert marks["completed"] == "[✓]"
        assert marks["in_progress"] == "[→]"
        assert marks["blocked"] == "[!]"
        assert marks["not_started"] == "[ ]"


class TestPlan:
    def test_plan_creation(self):
        plan = Plan(
            plan_id="test-plan",
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"],
            step_statuses=["not_started", "not_started", "not_started"],
            step_notes=["", "", ""],
        )

        assert plan.plan_id == "test-plan"
        assert plan.title == "Test Plan"
        assert len(plan.steps) == 3
        assert plan.created_at is not None
        assert plan.updated_at is not None

    def test_plan_with_metadata(self):
        metadata = {"priority": "high", "assignee": "agent_1"}
        plan = Plan(
            plan_id="test-plan",
            title="Test Plan",
            steps=["Step 1"],
            step_statuses=["not_started"],
            step_notes=[""],
            metadata=metadata,
        )

        assert plan.metadata["priority"] == "high"
        assert plan.metadata["assignee"] == "agent_1"


class TestPlanningTool:
    def test_tool_properties(self):
        tool = PlanningTool()
        assert tool.name == "planning"
        assert tool.category == "planning"
        assert "planning" in tool.tags
        assert "workflow" in tool.tags
        assert "command" in tool.required_parameters

    def test_parameters_schema(self):
        tool = PlanningTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "command" in params["properties"]
        assert "plan_id" in params["properties"]
        assert "title" in params["properties"]
        assert "steps" in params["properties"]

        commands = params["properties"]["command"]["enum"]
        assert "create" in commands
        assert "update" in commands
        assert "list" in commands
        assert "get" in commands
        assert "set_active" in commands
        assert "mark_step" in commands
        assert "delete" in commands

    def test_missing_command(self):
        tool = PlanningTool()
        result = tool.execute()
        assert result.error is not None
        assert "command" in result.error.lower()

    def test_unknown_command(self):
        tool = PlanningTool()
        result = tool.execute(command="unknown")
        assert result.error is not None
        assert "Unrecognized command" in result.error

    def test_create_plan_success(self):
        tool = PlanningTool()
        result = tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"],
        )

        assert result.error is None
        assert "test-plan-1" in result.output
        assert result.output is not None

    def test_create_plan_missing_plan_id(self):
        tool = PlanningTool()
        result = tool.execute(
            command="create",
            title="Test Plan",
            steps=["Step 1"],
        )

        assert result.error is not None
        assert "plan_id" in result.error

    def test_create_plan_missing_title(self):
        tool = PlanningTool()
        result = tool.execute(
            command="create",
            plan_id="test-plan-1",
            steps=["Step 1"],
        )

        assert result.error is not None
        assert "title" in result.error

    def test_create_plan_missing_steps(self):
        tool = PlanningTool()
        result = tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
        )

        assert result.error is not None
        assert "steps" in result.error

    def test_create_plan_invalid_steps(self):
        tool = PlanningTool()
        result = tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps="not a list",
        )

        assert result.error is not None
        assert "steps" in result.error

    def test_create_duplicate_plan(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1"],
        )

        result = tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Another Plan",
            steps=["Step 1"],
        )

        assert result.error is not None
        assert "already exists" in result.error

    def test_update_plan_success(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2"],
        )

        result = tool.execute(
            command="update",
            plan_id="test-plan-1",
            title="Updated Plan",
            steps=["Step 1", "Step 2", "Step 3"],
        )

        assert result.error is None
        assert "Updated Plan" in result.output

    def test_update_nonexistent_plan(self):
        tool = PlanningTool()
        result = tool.execute(
            command="update",
            plan_id="nonexistent",
            title="Updated Plan",
        )

        assert result.error is not None
        assert "No plan found" in result.error

    def test_list_plans_empty(self):
        tool = PlanningTool()
        result = tool.execute(command="list")

        assert result.error is None
        assert "No plans available" in result.output

    def test_list_plans_with_plans(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="plan-1",
            title="Plan 1",
            steps=["Step 1"],
        )
        tool.execute(
            command="create",
            plan_id="plan-2",
            title="Plan 2",
            steps=["Step 1", "Step 2"],
        )

        result = tool.execute(command="list")

        assert result.error is None
        assert "plan-1" in result.output
        assert "plan-2" in result.output

    def test_get_plan_success(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1"],
        )

        result = tool.execute(command="get", plan_id="test-plan-1")

        assert result.error is None
        assert "Test Plan" in result.output

    def test_get_plan_nonexistent(self):
        tool = PlanningTool()
        result = tool.execute(command="get", plan_id="nonexistent")

        assert result.error is not None
        assert "No plan found" in result.error

    def test_get_active_plan(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1"],
        )

        result = tool.execute(command="get")

        assert result.error is None
        assert "test-plan-1" in result.output

    def test_set_active_plan(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="plan-1",
            title="Plan 1",
            steps=["Step 1"],
        )
        tool.execute(
            command="create",
            plan_id="plan-2",
            title="Plan 2",
            steps=["Step 1"],
        )

        result = tool.execute(command="set_active", plan_id="plan-1")

        assert result.error is None
        assert "active plan" in result.output.lower()

        active = tool.get_active_plan()
        assert active.plan_id == "plan-1"

    def test_mark_step_success(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"],
        )

        result = tool.execute(
            command="mark_step",
            plan_id="test-plan-1",
            step_index=0,
            step_status="completed",
            step_notes="Done successfully",
        )

        assert result.error is None
        assert "completed" in result.output

    def test_mark_step_invalid_index(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1"],
        )

        result = tool.execute(
            command="mark_step",
            plan_id="test-plan-1",
            step_index=10,
            step_status="completed",
        )

        assert result.error is not None
        assert "Invalid step_index" in result.error

    def test_mark_step_invalid_status(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1"],
        )

        result = tool.execute(
            command="mark_step",
            plan_id="test-plan-1",
            step_index=0,
            step_status="invalid_status",
        )

        assert result.error is not None
        assert "Invalid step_status" in result.error

    def test_delete_plan(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1"],
        )

        result = tool.execute(command="delete", plan_id="test-plan-1")

        assert result.error is None
        assert "deleted" in result.output

        result = tool.execute(command="get", plan_id="test-plan-1")
        assert result.error is not None

    def test_delete_nonexistent_plan(self):
        tool = PlanningTool()
        result = tool.execute(command="delete", plan_id="nonexistent")

        assert result.error is not None
        assert "No plan found" in result.error

    def test_get_active_plan_method(self):
        tool = PlanningTool()
        assert tool.get_active_plan() is None

        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2"],
        )

        active = tool.get_active_plan()
        assert active is not None
        assert active.plan_id == "test-plan-1"

    def test_get_next_step(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"],
        )

        next_step = tool.get_next_step()
        assert next_step is not None
        assert next_step[0] == 0
        assert next_step[1] == "Step 1"

        tool.execute(
            command="mark_step",
            step_index=0,
            step_status="completed",
        )

        next_step = tool.get_next_step()
        assert next_step[0] == 1
        assert next_step[1] == "Step 2"

    def test_get_progress(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3", "Step 4"],
        )

        tool.execute(command="mark_step", step_index=0, step_status="completed")
        tool.execute(command="mark_step", step_index=1, step_status="in_progress")

        progress = tool.get_progress()

        assert progress["plan_id"] == "test-plan-1"
        assert progress["total_steps"] == 4
        assert progress["completed"] == 1
        assert progress["in_progress"] == 1
        assert progress["progress_percentage"] == 25.0

    def test_format_plan(self):
        tool = PlanningTool()
        tool.execute(
            command="create",
            plan_id="test-plan-1",
            title="Test Plan",
            steps=["Step 1", "Step 2"],
        )

        plan = tool.get_active_plan()
        formatted = tool._format_plan(plan)

        assert "Test Plan" in formatted
        assert "test-plan-1" in formatted
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "Progress:" in formatted
