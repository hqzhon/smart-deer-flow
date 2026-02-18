// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { resolveServiceURL } from "./resolve-service-url";

export interface ChartSeries {
  name: string;
  data: Array<number>;
  color?: string;
}

export interface ChartRequest {
  chart_type: "line" | "bar" | "pie" | "scatter" | "histogram" | "area";
  title?: string;
  labels?: string[];
  data?: Array<number>;
  series?: ChartSeries[];
  x_label?: string;
  y_label?: string;
  options?: Record<string, unknown>;
}

export interface ChartResponse {
  chart_type: string;
  title?: string;
  config: Record<string, unknown>;
  chart_data: Record<string, unknown>;
}

export interface StatisticsResponse {
  count: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  range: number;
  variance: number;
  sum: number;
  quartiles?: {
    q1: number;
    q2: number;
    q3: number;
    iqr: number;
  };
  percentiles?: Record<string, number>;
}

export interface CorrelationResponse {
  pearson: number;
  spearman?: number;
  sample_size: number;
  interpretation: string;
}

export interface DataAnalysisResponse {
  summary: Record<string, unknown>;
  statistics?: Record<string, StatisticsResponse>;
  correlations?: Record<string, CorrelationResponse>;
  recommendations: string[];
}

export async function createChart(request: ChartRequest): Promise<ChartResponse> {
  const response = await fetch(resolveServiceURL("analysis/chart"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(`Failed to create chart: ${response.statusText}`);
  }
  return response.json();
}

export async function calculateStatistics(
  data: Array<number>,
  includePercentiles = true
): Promise<StatisticsResponse> {
  const response = await fetch(resolveServiceURL("analysis/statistics"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data, include_percentiles: includePercentiles }),
  });
  if (!response.ok) {
    throw new Error(`Failed to calculate statistics: ${response.statusText}`);
  }
  return response.json();
}

export async function calculateCorrelation(
  xData: Array<number>,
  yData: Array<number>
): Promise<CorrelationResponse> {
  const response = await fetch(resolveServiceURL("analysis/correlation"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x_data: xData, y_data: yData }),
  });
  if (!response.ok) {
    throw new Error(`Failed to calculate correlation: ${response.statusText}`);
  }
  return response.json();
}

export async function analyzeData(
  data: Array<Record<string, unknown>>,
  columns?: string[],
  analysisType: "summary" | "distribution" | "correlation" = "summary"
): Promise<DataAnalysisResponse> {
  const response = await fetch(resolveServiceURL("analysis/analyze"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data,
      columns,
      analysis_type: analysisType,
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to analyze data: ${response.statusText}`);
  }
  return response.json();
}
