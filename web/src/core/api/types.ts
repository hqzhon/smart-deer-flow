// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import type { Option } from "../messages";

// Tool Calls

export interface ToolCall {
  type: "tool_call";
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolCallChunk {
  type: "tool_call_chunk";
  index: number;
  id: string;
  name: string;
  args: string;
}

// Events

interface GenericEvent<T extends string, D extends object> {
  type: T;
  data: {
    id: string;
    thread_id: string;
    agent: "coordinator" | "planner" | "researcher" | "coder" | "reporter" | "unknown";
    role: "user" | "assistant" | "tool";
    finish_reason?: "stop" | "tool_calls" | "interrupt";
  } & D;
}

export interface MessageChunkEvent
  extends GenericEvent<
    "message_chunk",
    {
      content?: string;
      reasoning_content?: string;
    }
  > {}

export interface ToolCallsEvent
  extends GenericEvent<
    "tool_calls",
    {
      tool_calls: ToolCall[];
      tool_call_chunks: ToolCallChunk[];
    }
  > {}

export interface ToolCallChunksEvent
  extends GenericEvent<
    "tool_call_chunks",
    {
      tool_call_chunks: ToolCallChunk[];
    }
  > {}

export interface ToolCallResultEvent
  extends GenericEvent<
    "tool_call_result",
    {
      tool_call_id: string;
      content?: string;
    }
  > {}

export interface InterruptEvent
  extends GenericEvent<
    "interrupt",
    {
      options: Option[];
    }
  > {}

export interface ErrorEvent {
  type: "error";
  data: {
    thread_id: string;
    error?: string;
    message?: string;
  };
}

// Structured Data Events

export interface ReflectionInsightsEvent {
  type: 'reflection_insights'
  data: {
    node: string
    insights: {
      // comprehensive_report field removed - reports are now generated separately
      knowledge_gaps?: string[]
      primary_follow_up_query?: string
      confidence_score?: number
    }
  }
}

export interface IsolationMetricsEvent {
  type: 'isolation_metrics'
  data: {
    node: string
    metrics: {
      compression_ratio?: number
      token_savings?: number
      performance_impact?: number
      memory_usage?: number
    }
  }
}

export interface ResearchProgressEvent {
  type: 'research_progress'
  data: {
    node: string
    progress: {
      current_step?: string
      completion_percentage?: number
      estimated_time?: string
      status?: string
    }
  }
}

export interface NodeUpdateEvent {
  type: 'node_update'
  data: {
    node: string
    agent: string
    data: Record<string, unknown>
    timestamp: number
  }
  metadata: Record<string, unknown>
}

export type ChatEvent =
  | MessageChunkEvent
  | ToolCallsEvent
  | ToolCallChunksEvent
  | ToolCallResultEvent
  | InterruptEvent
  | ErrorEvent
  | ReflectionInsightsEvent
  | IsolationMetricsEvent
  | ResearchProgressEvent
  | NodeUpdateEvent;
