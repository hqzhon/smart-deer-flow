# SPDX-License-Identifier: MIT
"""
Workflow Integration Package
Provides workflow integration capabilities for enhanced reflection
"""

from .reflection_workflow import ReflectionWorkflow, WorkflowStage, WorkflowResult

# Import the main workflow function from the parent workflow.py file
try:
    # Import from the parent src directory
    import importlib.util
    import os

    # Get the path to the workflow.py file in the parent src directory
    workflow_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "workflow.py"
    )

    if os.path.exists(workflow_path):
        spec = importlib.util.spec_from_file_location("workflow_module", workflow_path)
        workflow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow_module)
        run_agent_workflow_async = workflow_module.run_agent_workflow_async
    else:
        raise ImportError("workflow.py not found")
except Exception as import_error:
    # If import fails, define a placeholder
    def run_agent_workflow_async(*args, **kwargs):
        raise NotImplementedError(
            f"run_agent_workflow_async not available: {import_error}"
        )

    # Store the error for later use
    _import_error = import_error


__all__ = [
    "ReflectionWorkflow",
    "WorkflowStage",
    "WorkflowResult",
    "run_agent_workflow_async",
]
