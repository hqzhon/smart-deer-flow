from src.tools.visualization.data_visualization import (
    DataVisualizationTool,
    DataAnalysisTool,
)


class TestDataVisualizationTool:
    def test_tool_properties(self):
        tool = DataVisualizationTool()
        assert tool.name == "data_visualization"
        assert tool.category == "visualization"
        assert "visualization" in tool.tags
        assert "chart_type" in tool.required_parameters
        assert "data" in tool.required_parameters

    def test_parameters_schema(self):
        tool = DataVisualizationTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "chart_type" in params["properties"]
        assert "data" in params["properties"]
        assert "title" in params["properties"]

        chart_types = params["properties"]["chart_type"]["enum"]
        assert "line" in chart_types
        assert "bar" in chart_types
        assert "pie" in chart_types
        assert "scatter" in chart_types
        assert "histogram" in chart_types


class TestDataAnalysisTool:
    def test_tool_properties(self):
        tool = DataAnalysisTool()
        assert tool.name == "data_analysis"
        assert tool.category == "analysis"
        assert "analysis" in tool.tags
        assert "data" in tool.required_parameters

    def test_parameters_schema(self):
        tool = DataAnalysisTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "data" in params["properties"]
        assert "analysis_type" in params["properties"]

        analysis_types = params["properties"]["analysis_type"]["enum"]
        assert "describe" in analysis_types
        assert "correlation" in analysis_types
        assert "outliers" in analysis_types
        assert "distribution" in analysis_types
        assert "summary" in analysis_types

    def test_unknown_analysis_type(self):
        tool = DataAnalysisTool()

        data = {"values": [1, 2, 3]}
        result = tool.execute(data=data, analysis_type="unknown")

        assert result.error is not None
        assert "Unknown analysis type" in result.error
