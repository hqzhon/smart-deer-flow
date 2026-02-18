import asyncio
import logging
import os
import tempfile
from typing import Any, Optional

from src.tools.base_tool import BaseTool
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)

_VISUALIZATION_DESCRIPTION = """\
A tool for data visualization and chart generation.

This tool supports:
1. Creating various chart types (line, bar, pie, scatter, histogram, etc.)
2. Generating charts from data arrays or CSV files
3. Saving charts as PNG or HTML files
4. Adding statistical insights to charts

Use this when you need to visualize data, create reports with charts, or analyze data patterns.
"""


class DataVisualizationTool(BaseTool):
    name: str = "data_visualization"
    description: str = _VISUALIZATION_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": [
                        "line",
                        "bar",
                        "pie",
                        "scatter",
                        "histogram",
                        "box",
                        "area",
                        "heatmap",
                    ],
                    "description": "Type of chart to create",
                },
                "data": {
                    "type": "object",
                    "description": "Data for the chart. Format: {'x': [...], 'y': [...]} or {'labels': [...], 'values': [...]}",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title",
                },
                "x_label": {
                    "type": "string",
                    "description": "X-axis label",
                },
                "y_label": {
                    "type": "string",
                    "description": "Y-axis label",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "html", "base64"],
                    "default": "base64",
                    "description": "Output format for the chart",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the chart (optional)",
                },
                "options": {
                    "type": "object",
                    "description": "Additional chart options (colors, styles, etc.)",
                },
            },
            "required": ["chart_type", "data"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["chart_type", "data"]

    @property
    def category(self) -> str:
        return "visualization"

    @property
    def tags(self) -> list[str]:
        return ["visualization", "chart", "data", "analysis"]

    def execute(self, **kwargs) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.async_execute(**kwargs))

    async def async_execute(
        self,
        chart_type: str,
        data: dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        output_format: str = "base64",
        output_path: Optional[str] = None,
        options: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import numpy as np

            matplotlib.use("Agg")

            options = options or {}
            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            x_data = data.get("x", data.get("labels", []))
            y_data = data.get("y", data.get("values", []))

            if chart_type == "line":
                ax.plot(
                    x_data,
                    y_data,
                    marker=options.get("marker", "o"),
                    color=options.get("color", "blue"),
                )
            elif chart_type == "bar":
                colors = options.get("colors", plt.cm.tab10.colors[: len(x_data)])
                ax.bar(x_data, y_data, color=colors)
            elif chart_type == "pie":
                colors = options.get("colors", plt.cm.Set3.colors[: len(y_data)])
                ax.pie(
                    y_data,
                    labels=x_data,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                )
            elif chart_type == "scatter":
                ax.scatter(x_data, y_data, c=options.get("color", "blue"), alpha=0.6)
            elif chart_type == "histogram":
                ax.hist(
                    y_data if y_data else x_data,
                    bins=options.get("bins", 10),
                    color=options.get("color", "steelblue"),
                    edgecolor="white",
                )
            elif chart_type == "box":
                ax.boxplot(
                    y_data if isinstance(y_data[0], list) else [y_data],
                    labels=x_data if x_data else ["Data"],
                )
            elif chart_type == "area":
                ax.fill_between(
                    x_data, y_data, alpha=0.4, color=options.get("color", "blue")
                )
                ax.plot(x_data, y_data, color=options.get("color", "blue"))
            elif chart_type == "heatmap":
                import numpy as np

                z_data = data.get("z", np.random.rand(len(x_data), len(y_data)))
                im = ax.imshow(z_data, cmap=options.get("cmap", "viridis"))
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(x_data)))
                ax.set_yticks(range(len(y_data)))
                ax.set_xticklabels(x_data)
                ax.set_yticklabels(y_data)

            if title:
                ax.set_title(title, fontsize=14, fontweight="bold")
            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)

            if chart_type != "pie":
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_format == "base64":
                import base64
                from io import BytesIO

                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plt.close(fig)
                return ToolResult(
                    output=f"Chart '{chart_type}' created successfully",
                    base64_image=img_base64,
                )
            elif output_format == "html":
                import plotly.io as pio

                fig_plotly = self._convert_to_plotly(
                    chart_type, data, title, x_label, y_label, options
                )
                html_content = pio.to_html(fig_plotly, full_html=True)
                if output_path:
                    with open(output_path, "w") as f:
                        f.write(html_content)
                    plt.close(fig)
                    return ToolResult(output=f"Chart saved to {output_path}")
                plt.close(fig)
                return ToolResult(
                    output=f"Chart created as HTML:\n{html_content[:500]}..."
                )
            else:
                if not output_path:
                    output_path = os.path.join(
                        tempfile.gettempdir(), f"chart_{chart_type}.png"
                    )
                plt.savefig(output_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                return ToolResult(output=f"Chart saved to {output_path}")

        except ImportError as e:
            return ToolResult(
                error=f"Required library not installed: {e}. Install with: pip install matplotlib numpy"
            )
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return ToolResult(error=f"Chart generation failed: {str(e)}")

    def _convert_to_plotly(
        self,
        chart_type: str,
        data: dict,
        title: Optional[str],
        x_label: Optional[str],
        y_label: Optional[str],
        options: Optional[dict],
    ) -> Any:
        import plotly.graph_objects as go

        x_data = data.get("x", data.get("labels", []))
        y_data = data.get("y", data.get("values", []))
        options = options or {}

        if chart_type == "line":
            fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode="lines+markers"))
        elif chart_type == "bar":
            fig = go.Figure(data=go.Bar(x=x_data, y=y_data))
        elif chart_type == "pie":
            fig = go.Figure(data=go.Pie(labels=x_data, values=y_data))
        elif chart_type == "scatter":
            fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode="markers"))
        elif chart_type == "histogram":
            fig = go.Figure(
                data=go.Histogram(
                    x=y_data if y_data else x_data, nbinsx=options.get("bins", 10)
                )
            )
        elif chart_type == "area":
            fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, fill="tozeroy"))
        else:
            fig = go.Figure(data=go.Scatter(x=x_data, y=y_data))

        fig.update_layout(
            title=title or "",
            xaxis_title=x_label or "",
            yaxis_title=y_label or "",
        )
        return fig


class DataAnalysisTool(BaseTool):
    name: str = "data_analysis"
    description: str = """Analyze data and generate statistical insights.

This tool can:
1. Calculate descriptive statistics (mean, median, std, etc.)
2. Detect patterns and correlations
3. Identify outliers
4. Generate summary reports

Input data can be:
- A list of numbers
- A dictionary with column names and values
- A path to a CSV file
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Data to analyze. Can be {'column': [values]} or {'values': [...] }",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": [
                        "describe",
                        "correlation",
                        "outliers",
                        "distribution",
                        "summary",
                    ],
                    "default": "describe",
                    "description": "Type of analysis to perform",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to analyze (optional)",
                },
            },
            "required": ["data"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["data"]

    @property
    def category(self) -> str:
        return "analysis"

    @property
    def tags(self) -> list[str]:
        return ["analysis", "statistics", "data"]

    def execute(
        self,
        data: dict[str, Any],
        analysis_type: str = "describe",
        columns: Optional[list[str]] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            import pandas as pd

            if "values" in data:
                df = pd.DataFrame({"values": data["values"]})
            else:
                df = pd.DataFrame(data)

            if columns:
                df = df[columns]

            if analysis_type == "describe":
                result = df.describe(include="all").to_dict()
                return ToolResult(
                    output=f"Descriptive statistics calculated:\n{result}"
                )

            elif analysis_type == "correlation":
                numeric_df = df.select_dtypes(include=["number"])
                if numeric_df.empty:
                    return ToolResult(
                        error="No numeric columns for correlation analysis"
                    )
                corr_matrix = numeric_df.corr().to_dict()
                return ToolResult(
                    output=f"Correlation matrix calculated:\n{corr_matrix}"
                )

            elif analysis_type == "outliers":
                import numpy as np

                numeric_df = df.select_dtypes(include=[np.number])
                outliers = {}
                for col in numeric_df.columns:
                    q1 = numeric_df[col].quantile(0.25)
                    q3 = numeric_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_indices = numeric_df[
                        (numeric_df[col] < lower) | (numeric_df[col] > upper)
                    ].index.tolist()
                    if outlier_indices:
                        outliers[col] = {
                            "count": len(outlier_indices),
                            "indices": outlier_indices[:10],
                            "bounds": {"lower": lower, "upper": upper},
                        }
                return ToolResult(
                    output=f"Found outliers in {len(outliers)} columns:\n{outliers}"
                )

            elif analysis_type == "distribution":
                distributions = {}
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        distributions[col] = {
                            "mean": float(df[col].mean()),
                            "median": float(df[col].median()),
                            "std": float(df[col].std()),
                            "skew": float(df[col].skew()),
                            "kurtosis": float(df[col].kurtosis()),
                        }
                return ToolResult(
                    output=f"Distribution analysis completed:\n{distributions}"
                )

            elif analysis_type == "summary":
                summary = {
                    "shape": {"rows": len(df), "columns": len(df.columns)},
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_values": df.isnull().sum().to_dict(),
                    "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                }
                return ToolResult(output=f"Data summary generated:\n{summary}")

            return ToolResult(error=f"Unknown analysis type: {analysis_type}")

        except ImportError as e:
            return ToolResult(
                error=f"Required library not installed: {e}. Install with: pip install pandas numpy"
            )
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return ToolResult(error=f"Data analysis failed: {str(e)}")
