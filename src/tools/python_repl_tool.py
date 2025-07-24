"""
Python REPL tool implementation using the new BaseTool interface.
Provides safe Python code execution capabilities.
"""

import logging
import sys
from io import StringIO
from typing import Any, Dict, List

from src.tools.base_tool import BaseTool, ToolInput, ToolOutput

logger = logging.getLogger(__name__)


class PythonREPLInput(ToolInput):
    """Input model for Python REPL tool."""

    code: str
    timeout: int = 30


class PythonREPLTool(BaseTool):
    """Safe Python code execution tool."""

    @property
    def name(self) -> str:
        return "python_repl"

    @property
    def description(self) -> str:
        return "Execute Python code safely in a restricted environment"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 30,
                "minimum": 1,
                "maximum": 300,
            },
        }

    @property
    def required_parameters(self) -> List[str]:
        return ["code"]

    @property
    def category(self) -> str:
        return "code"

    @property
    def tags(self) -> List[str]:
        return ["python", "repl", "code", "execution"]

    def execute(self, code: str, timeout: int = 30) -> ToolOutput:
        """Execute Python code safely.

        Args:
            code: Python code to execute
            timeout: Execution timeout

        Returns:
            ToolOutput with execution results
        """
        try:
            # Security check - block dangerous imports
            dangerous_imports = [
                "os.system",
                "subprocess",
                "eval",
                "exec",
                "compile",
                "__import__",
                "open",
                "file",
                "input",
                "raw_input",
                "socket",
                "urllib",
                "requests",
                "http.client",
            ]

            for dangerous in dangerous_imports:
                if dangerous in code.lower():
                    return ToolOutput(
                        success=False,
                        message=f"Security error: {dangerous} usage not allowed",
                        data={"error": "Security restriction", "code": code},
                    )

            # Create safe execution environment
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bin": bin,
                    "bool": bool,
                    "chr": chr,
                    "dict": dict,
                    "dir": dir,
                    "divmod": divmod,
                    "enumerate": enumerate,
                    "filter": filter,
                    "float": float,
                    "format": format,
                    "frozenset": frozenset,
                    "hash": hash,
                    "hex": hex,
                    "int": int,
                    "isinstance": isinstance,
                    "issubclass": issubclass,
                    "len": len,
                    "list": list,
                    "map": map,
                    "max": max,
                    "min": min,
                    "oct": oct,
                    "ord": ord,
                    "pow": pow,
                    "print": print,
                    "range": range,
                    "repr": repr,
                    "reversed": reversed,
                    "round": round,
                    "set": set,
                    "slice": slice,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "type": type,
                    "zip": zip,
                }
            }

            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Execute code
                compiled_code = compile(code, "<string>", "exec")
                exec(compiled_code, safe_globals)

                # Get results
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                result = {
                    "stdout": stdout_output,
                    "stderr": stderr_output,
                    "executed_code": code,
                }

                if stderr_output:
                    return ToolOutput(
                        success=False, message="Code executed with errors", data=result
                    )
                else:
                    return ToolOutput(
                        success=True, message="Code executed successfully", data=result
                    )

            except Exception as e:
                return ToolOutput(
                    success=False,
                    message=f"Execution error: {str(e)}",
                    data={
                        "error": str(e),
                        "executed_code": code,
                        "stdout": stdout_capture.getvalue(),
                        "stderr": stderr_capture.getvalue(),
                    },
                )

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            logger.error(f"Python REPL failed: {e}")
            return ToolOutput(
                success=False, message=f"REPL error: {str(e)}", data={"error": str(e)}
            )


class DataAnalysisTool(BaseTool):
    """Data analysis tool for processing datasets."""

    @property
    def name(self) -> str:
        return "analyze_data"

    @property
    def description(self) -> str:
        return "Analyze data using pandas and numpy"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "data": {
                "type": "array",
                "description": "Data to analyze (list of dictionaries or values)",
            },
            "operation": {
                "type": "string",
                "description": "Analysis operation to perform",
                "enum": ["mean", "median", "sum", "count", "describe", "correlation"],
            },
        }

    @property
    def required_parameters(self) -> List[str]:
        return ["data", "operation"]

    @property
    def category(self) -> str:
        return "analysis"

    @property
    def tags(self) -> List[str]:
        return ["data", "analysis", "pandas", "statistics"]

    def execute(self, data: List, operation: str) -> ToolOutput:
        """Execute data analysis.

        Args:
            data: Data to analyze
            operation: Analysis operation

        Returns:
            ToolOutput with analysis results
        """
        try:
            import pandas as pd

            # Convert data to DataFrame
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame({"values": data})

            # Perform analysis
            if operation == "mean":
                result = df.mean().to_dict()
            elif operation == "median":
                result = df.median().to_dict()
            elif operation == "sum":
                result = df.sum().to_dict()
            elif operation == "count":
                result = df.count().to_dict()
            elif operation == "describe":
                result = df.describe().to_dict()
            elif operation == "correlation":
                result = df.corr().to_dict()
            else:
                return ToolOutput(
                    success=False,
                    message=f"Unsupported operation: {operation}",
                    data={},
                )

            return ToolOutput(
                success=True,
                message=f"Data analysis completed: {operation}",
                data={"result": result, "operation": operation},
            )

        except ImportError:
            return ToolOutput(
                success=False,
                message="pandas and numpy not available for data analysis",
                data={},
            )
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return ToolOutput(
                success=False, message=f"Analysis failed: {str(e)}", data={}
            )
