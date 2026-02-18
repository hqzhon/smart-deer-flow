// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { resolveServiceURL } from "./resolve-service-url";

export interface SystemMetrics {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  memory_used: string;
  memory_total: string;
  disk_usage: number;
  disk_used: string;
  disk_total: string;
  network_connections: number;
  process_count: number;
  load_average: number[];
}

export interface LLMMetrics {
  total_requests: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  total_cost: number;
  average_latency: number;
  requests_per_minute: number;
  tokens_per_minute: number;
  error_rate: number;
  models_used: Record<string, number>;
}

export interface ContextMetrics {
  total_sessions: number;
  active_sessions: number;
  total_messages: number;
  total_context_size: number;
  average_context_size: number;
  compression_ratio: number;
  cache_hit_rate: number;
}

export interface PerformanceMetrics {
  request_queue_size: number;
  active_connections: number;
  max_connections: number;
  connection_utilization: number;
  average_response_time: number;
  p50_response_time: number;
  p95_response_time: number;
  p99_response_time: number;
}

export interface MetricsHistory {
  timestamps: string[];
  cpu_usage: number[];
  memory_usage: number[];
  request_count: number[];
  response_time: number[];
}

export interface MetricsDashboard {
  system: SystemMetrics;
  llm: LLMMetrics;
  context: ContextMetrics;
  performance: PerformanceMetrics;
  uptime: number;
  version: string;
}

export async function getSystemMetrics(): Promise<SystemMetrics> {
  const response = await fetch(resolveServiceURL("metrics/system"));
  if (!response.ok) {
    throw new Error(`Failed to get system metrics: ${response.statusText}`);
  }
  return response.json();
}

export async function getLLMMetrics(): Promise<LLMMetrics> {
  const response = await fetch(resolveServiceURL("metrics/llm"));
  if (!response.ok) {
    throw new Error(`Failed to get LLM metrics: ${response.statusText}`);
  }
  return response.json();
}

export async function getContextMetrics(): Promise<ContextMetrics> {
  const response = await fetch(resolveServiceURL("metrics/context"));
  if (!response.ok) {
    throw new Error(`Failed to get context metrics: ${response.statusText}`);
  }
  return response.json();
}

export async function getPerformanceMetrics(): Promise<PerformanceMetrics> {
  const response = await fetch(resolveServiceURL("metrics/performance"));
  if (!response.ok) {
    throw new Error(`Failed to get performance metrics: ${response.statusText}`);
  }
  return response.json();
}

export async function getMetricsHistory(
  duration = 3600
): Promise<MetricsHistory> {
  const response = await fetch(
    `${resolveServiceURL("metrics/history")}?duration=${duration}`
  );
  if (!response.ok) {
    throw new Error(`Failed to get metrics history: ${response.statusText}`);
  }
  return response.json();
}

export async function getMetricsDashboard(): Promise<MetricsDashboard> {
  const response = await fetch(resolveServiceURL("metrics/dashboard"));
  if (!response.ok) {
    throw new Error(`Failed to get metrics dashboard: ${response.statusText}`);
  }
  return response.json();
}

export async function recordMetric(
  metricType: string,
  value: number,
  metadata?: Record<string, unknown>
): Promise<{ status: string }> {
  const response = await fetch(
    `${resolveServiceURL("metrics/record")}?metric_type=${metricType}&value=${value}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata }),
    }
  );
  if (!response.ok) {
    throw new Error(`Failed to record metric: ${response.statusText}`);
  }
  return response.json();
}

export async function recordLLMUsage(data: {
  model: string;
  input_tokens: number;
  output_tokens: number;
  latency: number;
  cost?: number;
  error?: boolean;
}): Promise<{ status: string }> {
  const response = await fetch(resolveServiceURL("metrics/llm/record"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Failed to record LLM usage: ${response.statusText}`);
  }
  return response.json();
}
