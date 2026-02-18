# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class ChartDataPoint(BaseModel):
    label: str
    value: Union[float, int]
    color: Optional[str] = None


class ChartSeries(BaseModel):
    name: str
    data: List[Union[float, int]]
    color: Optional[str] = None


class ChartRequest(BaseModel):
    chart_type: str = Field(
        ...,
        description="Type of chart: line, bar, pie, scatter, histogram, area",
    )
    title: Optional[str] = Field(None, description="Chart title")
    labels: Optional[List[str]] = Field(None, description="Labels for data points")
    data: Optional[List[Union[float, int]]] = Field(
        None, description="Single series data"
    )
    series: Optional[List[ChartSeries]] = Field(
        None, description="Multiple series data"
    )
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional chart options"
    )


class ChartResponse(BaseModel):
    chart_type: str
    title: Optional[str]
    config: Dict[str, Any]
    chart_data: Dict[str, Any]


class StatisticsRequest(BaseModel):
    data: List[Union[float, int]] = Field(
        ..., description="Data for statistical analysis"
    )
    include_percentiles: bool = Field(
        True, description="Include percentile calculations"
    )


class StatisticsResponse(BaseModel):
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    range: float
    variance: float
    sum: float
    quartiles: Optional[Dict[str, float]] = None
    percentiles: Optional[Dict[str, float]] = None


class CorrelationRequest(BaseModel):
    x_data: List[Union[float, int]] = Field(..., description="First variable data")
    y_data: List[Union[float, int]] = Field(..., description="Second variable data")


class CorrelationResponse(BaseModel):
    pearson: float
    spearman: Optional[float] = None
    sample_size: int
    interpretation: str


class DataAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Data records for analysis")
    columns: Optional[List[str]] = Field(None, description="Columns to analyze")
    analysis_type: str = Field(
        "summary",
        description="Type of analysis: summary, distribution, correlation",
    )


class DataAnalysisResponse(BaseModel):
    summary: Dict[str, Any]
    statistics: Optional[Dict[str, StatisticsResponse]] = None
    correlations: Optional[Dict[str, CorrelationResponse]] = None
    recommendations: List[str] = Field(default_factory=list)


@router.post("/chart", response_model=ChartResponse)
async def create_chart(request: ChartRequest):
    """Create a chart configuration from data."""
    valid_types = ["line", "bar", "pie", "scatter", "histogram", "area"]
    if request.chart_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chart type. Must be one of: {valid_types}",
        )

    config = _build_chart_config(request)
    chart_data = _prepare_chart_data(request)

    return ChartResponse(
        chart_type=request.chart_type,
        title=request.title,
        config=config,
        chart_data=chart_data,
    )


@router.post("/statistics", response_model=StatisticsResponse)
async def calculate_statistics(request: StatisticsRequest):
    """Calculate statistical measures for data."""
    if not request.data:
        raise HTTPException(status_code=400, detail="Data cannot be empty")

    data = [float(x) for x in request.data]
    n = len(data)

    sorted_data = sorted(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = variance**0.5

    median = (
        sorted_data[n // 2]
        if n % 2 == 1
        else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    )

    quartiles = None
    percentiles = None

    if request.include_percentiles:
        q1_idx = int(n * 0.25)
        q3_idx = int(n * 0.75)
        quartiles = {
            "q1": sorted_data[q1_idx],
            "q2": median,
            "q3": sorted_data[q3_idx],
            "iqr": sorted_data[q3_idx] - sorted_data[q1_idx],
        }
        percentiles = {
            "p10": sorted_data[int(n * 0.10)],
            "p25": quartiles["q1"],
            "p50": median,
            "p75": quartiles["q3"],
            "p90": sorted_data[int(n * 0.90)],
            "p95": sorted_data[int(n * 0.95)],
            "p99": sorted_data[int(n * 0.99)],
        }

    return StatisticsResponse(
        count=n,
        mean=mean,
        median=median,
        std=std,
        min=min(data),
        max=max(data),
        range=max(data) - min(data),
        variance=variance,
        sum=sum(data),
        quartiles=quartiles,
        percentiles=percentiles,
    )


@router.post("/correlation", response_model=CorrelationResponse)
async def calculate_correlation(request: CorrelationRequest):
    """Calculate correlation between two variables."""
    if len(request.x_data) != len(request.y_data):
        raise HTTPException(
            status_code=400, detail="X and Y data must have the same length"
        )

    if len(request.x_data) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 data points are required"
        )

    x = [float(v) for v in request.x_data]
    y = [float(v) for v in request.y_data]
    n = len(x)

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = (
        sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)
    ) ** 0.5

    pearson = numerator / denominator if denominator != 0 else 0

    interpretation = _interpret_correlation(pearson)

    return CorrelationResponse(
        pearson=pearson,
        sample_size=n,
        interpretation=interpretation,
    )


@router.post("/analyze", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """Perform comprehensive data analysis."""
    if not request.data:
        raise HTTPException(status_code=400, detail="Data cannot be empty")

    columns = request.columns or list(request.data[0].keys()) if request.data else []

    summary = {
        "total_records": len(request.data),
        "columns": columns,
        "column_types": {},
    }

    for col in columns:
        values = [row.get(col) for row in request.data if col in row]
        if values:
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                summary["column_types"][col] = "numeric"
            elif all(isinstance(v, str) for v in values if v is not None):
                summary["column_types"][col] = "categorical"
            else:
                summary["column_types"][col] = "mixed"

    statistics = None
    correlations = None
    recommendations = []

    if request.analysis_type in ["summary", "distribution"]:
        statistics = {}
        for col in columns:
            values = [
                float(row[col])
                for row in request.data
                if col in row and isinstance(row[col], (int, float))
            ]
            if values:
                stats = await calculate_statistics(
                    StatisticsRequest(data=values, include_percentiles=True)
                )
                statistics[col] = stats

    if request.analysis_type == "correlation":
        numeric_cols = [
            col for col in columns if summary["column_types"].get(col) == "numeric"
        ]
        if len(numeric_cols) >= 2:
            correlations = {}
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    x_data = [
                        float(row[col1])
                        for row in request.data
                        if col1 in row and col2 in row
                    ]
                    y_data = [
                        float(row[col2])
                        for row in request.data
                        if col1 in row and col2 in row
                    ]
                    if x_data and y_data:
                        corr = await calculate_correlation(
                            CorrelationRequest(x_data=x_data, y_data=y_data)
                        )
                        correlations[f"{col1}_vs_{col2}"] = corr

    if statistics:
        for col, stats in statistics.items():
            if stats.std > stats.mean * 0.5:
                recommendations.append(
                    f"Column '{col}' has high variability (std/mean > 0.5)"
                )
            if stats.quartiles and stats.quartiles["iqr"] > stats.range * 0.5:
                recommendations.append(f"Column '{col}' has a wide interquartile range")

    return DataAnalysisResponse(
        summary=summary,
        statistics=statistics,
        correlations=correlations,
        recommendations=recommendations,
    )


def _build_chart_config(request: ChartRequest) -> Dict[str, Any]:
    """Build chart.js compatible configuration."""
    config = {
        "type": request.chart_type,
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": bool(request.title),
                    "text": request.title or "",
                },
                "legend": {
                    "display": True,
                    "position": "top",
                },
            },
        },
    }

    if request.chart_type in ["line", "bar", "scatter", "area"]:
        config["options"]["scales"] = {
            "x": {
                "title": {
                    "display": bool(request.x_label),
                    "text": request.x_label or "",
                }
            },
            "y": {
                "title": {
                    "display": bool(request.y_label),
                    "text": request.y_label or "",
                }
            },
        }

    if request.options:
        config["options"] = _deep_merge(config["options"], request.options)

    return config


def _prepare_chart_data(request: ChartRequest) -> Dict[str, Any]:
    """Prepare data for chart rendering."""
    chart_data = {}

    if request.data:
        chart_data["labels"] = request.labels or [
            str(i) for i in range(len(request.data))
        ]
        chart_data["datasets"] = [
            {
                "label": request.title or "Dataset",
                "data": request.data,
                "borderColor": "#3b82f6",
                "backgroundColor": "rgba(59, 130, 246, 0.5)",
            }
        ]
    elif request.series:
        chart_data["labels"] = request.labels or []
        chart_data["datasets"] = [
            {
                "label": s.name,
                "data": s.data,
                "borderColor": s.color or _get_default_color(i),
                "backgroundColor": f"{s.color or _get_default_color(i)}80",
            }
            for i, s in enumerate(request.series)
        ]

    return chart_data


def _get_default_color(index: int) -> str:
    """Get a default color from a palette."""
    colors = [
        "#3b82f6",
        "#ef4444",
        "#22c55e",
        "#f59e0b",
        "#8b5cf6",
        "#ec4899",
        "#06b6d4",
        "#84cc16",
    ]
    return colors[index % len(colors)]


def _interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient."""
    r_abs = abs(r)
    if r_abs >= 0.9:
        strength = "very strong"
    elif r_abs >= 0.7:
        strength = "strong"
    elif r_abs >= 0.5:
        strength = "moderate"
    elif r_abs >= 0.3:
        strength = "weak"
    else:
        strength = "very weak or no"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction} correlation"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
