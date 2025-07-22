// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import type {
  ChatEvent,
  InterruptEvent,
  MessageChunkEvent,
  ToolCallChunksEvent,
  ToolCallResultEvent,
  ToolCallsEvent,
  ReflectionInsightsEvent,
  IsolationMetricsEvent,
  ResearchProgressEvent,
  NodeUpdateEvent,
} from "../api";
import { deepClone } from "../utils/deep-clone";

import type { Message } from "./types";

export function mergeMessage(message: Message, event: ChatEvent) {
  if (event.type === "message_chunk") {
    mergeTextMessage(message, event);
  } else if (event.type === "tool_calls" || event.type === "tool_call_chunks") {
    mergeToolCallMessage(message, event);
  } else if (event.type === "tool_call_result") {
    mergeToolCallResultMessage(message, event);
  } else if (event.type === "interrupt") {
    mergeInterruptMessage(message, event);
  } else if (event.type === "reflection_insights") {
    mergeReflectionInsightsMessage(message, event);
  } else if (event.type === "isolation_metrics") {
    mergeIsolationMetricsMessage(message, event);
  } else if (event.type === "research_progress") {
    mergeResearchProgressMessage(message, event);
  } else if (event.type === "node_update") {
    mergeNodeUpdateMessage(message, event);
  }
  if ('finish_reason' in event.data && event.data.finish_reason) {
    message.finishReason = event.data.finish_reason;
    message.isStreaming = false;
    if (message.toolCalls) {
      message.toolCalls.forEach((toolCall) => {
        if (toolCall.argsChunks?.length) {
          const argsString = toolCall.argsChunks.join("");
          try {
            toolCall.args = JSON.parse(argsString);
            delete toolCall.argsChunks;
          } catch (parseError) {
            console.error('Failed to parse tool call args:', argsString, parseError);
            // Keep empty object as fallback to maintain type compatibility
            toolCall.args = {};
            delete toolCall.argsChunks;
          }
        }
      });
    }
  }
  return deepClone(message);
}

function mergeTextMessage(message: Message, event: MessageChunkEvent) {
  if (event.data.content) {
    message.content += event.data.content;
    message.contentChunks.push(event.data.content);
  }
  if (event.data.reasoning_content) {
    message.reasoningContent = (message.reasoningContent ?? "") + event.data.reasoning_content;
    message.reasoningContentChunks = message.reasoningContentChunks ?? [];
    message.reasoningContentChunks.push(event.data.reasoning_content);
  }
}

function mergeToolCallMessage(
  message: Message,
  event: ToolCallsEvent | ToolCallChunksEvent,
) {
  if (event.type === "tool_calls" && event.data.tool_calls[0]?.name) {
    message.toolCalls = event.data.tool_calls.map((raw) => ({
      id: raw.id,
      name: raw.name,
      args: raw.args,
      result: undefined,
    }));
  }

  message.toolCalls ??= [];
  for (const chunk of event.data.tool_call_chunks) {
    if (chunk.id) {
      const toolCall = message.toolCalls.find(
        (toolCall) => toolCall.id === chunk.id,
      );
      if (toolCall) {
        toolCall.argsChunks = [chunk.args];
      }
    } else {
      const streamingToolCall = message.toolCalls.find(
        (toolCall) => toolCall.argsChunks?.length,
      );
      if (streamingToolCall) {
        streamingToolCall.argsChunks!.push(chunk.args);
      }
    }
  }
}

function mergeToolCallResultMessage(
  message: Message,
  event: ToolCallResultEvent,
) {
  const toolCall = message.toolCalls?.find(
    (toolCall) => toolCall.id === event.data.tool_call_id,
  );
  if (toolCall) {
    toolCall.result = event.data.content;
  }
}

function mergeInterruptMessage(message: Message, event: InterruptEvent) {
  message.isStreaming = false;
  message.options = event.data.options;
}

function mergeReflectionInsightsMessage(
  message: Message,
  event: ReflectionInsightsEvent,
) {
  message.reflectionInsights = event.data.insights ?? {};
}

function mergeIsolationMetricsMessage(
  message: Message,
  event: IsolationMetricsEvent,
) {
  message.isolationMetrics = event.data.metrics;
}

function mergeResearchProgressMessage(
  message: Message,
  event: ResearchProgressEvent,
) {
  message.researchProgress = event.data.progress;
}

function mergeNodeUpdateMessage(
  message: Message,
  event: NodeUpdateEvent,
) {
  message.nodeUpdates = message.nodeUpdates ?? [];
  message.nodeUpdates.push({
    node: event.data.node,
    agent: event.data.agent,
    data: event.data.data,
    timestamp: event.data.timestamp,
  });
}
