// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

export type MessageRole = "user" | "assistant" | "tool";

export interface Message {
  id: string;
  threadId: string;
  agent?:
    | "coordinator"
    | "planner"
    | "researcher"
    | "coder"
    | "reporter"
    | "podcast";
  role: MessageRole;
  isStreaming?: boolean;
  content: string;
  contentChunks: string[];
  reasoningContent?: string;
  reasoningContentChunks?: string[];
  toolCalls?: ToolCallRuntime[];
  options?: Option[];
  finishReason?: "stop" | "interrupt" | "tool_calls";
  interruptFeedback?: string;
  resources?: Array<Resource>;
  // Structured Data
  reflectionInsights?: {
    comprehensive_report?: string;
    knowledge_gaps?: string[];
    primary_follow_up_query?: string;
    is_sufficient?: boolean;
    confidence_score?: number;
  };
  isolationMetrics?: {
    session_id?: string;
    compression_ratio?: number;
    token_savings?: number;
    performance_impact?: number;
    success_rate?: number;
    memory_usage?: number;
    [key: string]: unknown;
  };
  researchProgress?: {
    current_step?: string;
    completion_percentage?: number;
    estimated_time_remaining?: number;
    status?: string;
    [key: string]: unknown;
  };
  nodeUpdates?: Array<{
    node: string;
    agent: string;
    data: Record<string, unknown>;
    timestamp: number;
  }>;
}

export interface Option {
  text: string;
  value: string;
}

export interface ToolCallRuntime {
  id: string;
  name: string;
  args: Record<string, unknown>;
  argsChunks?: string[];
  result?: string;
}

export interface Resource {
  uri: string;
  title: string;
}
