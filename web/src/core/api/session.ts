// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { resolveServiceURL } from "./resolve-service-url";

export interface Session {
  id: string;
  thread_id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  status: "active" | "archived" | "deleted";
  metadata: Record<string, unknown>;
  message_count: number;
  last_message_preview?: string;
  research_topic?: string;
  report_style?: string;
  tags: string[];
}

export interface SessionDetail extends Session {
  messages: Array<Record<string, unknown>>;
  plan?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
}

export interface SessionCreate {
  title?: string;
  research_topic?: string;
  report_style?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface SessionUpdate {
  title?: string;
  status?: "active" | "archived" | "deleted";
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface SessionListResponse {
  sessions: Session[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export async function listSessions(params: {
  page?: number;
  page_size?: number;
  status?: string;
  tags?: string;
  search?: string;
  sort_by?: string;
  sort_order?: string;
} = {}): Promise<SessionListResponse> {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) {
      searchParams.append(key, String(value));
    }
  });

  const response = await fetch(
    `${resolveServiceURL("sessions")}?${searchParams.toString()}`
  );
  if (!response.ok) {
    throw new Error(`Failed to list sessions: ${response.statusText}`);
  }
  return response.json();
}

export async function createSession(data: SessionCreate): Promise<Session> {
  const response = await fetch(resolveServiceURL("sessions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.statusText}`);
  }
  return response.json();
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  const response = await fetch(resolveServiceURL(`sessions/${sessionId}`));
  if (!response.ok) {
    throw new Error(`Failed to get session: ${response.statusText}`);
  }
  return response.json();
}

export async function updateSession(
  sessionId: string,
  data: SessionUpdate
): Promise<Session> {
  const response = await fetch(resolveServiceURL(`sessions/${sessionId}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Failed to update session: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteSession(
  sessionId: string,
  hard = false
): Promise<{ status: string; session_id: string }> {
  const response = await fetch(
    `${resolveServiceURL(`sessions/${sessionId}`)}?hard=${hard}`,
    { method: "DELETE" }
  );
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
  return response.json();
}

export async function getSessionByThread(
  threadId: string
): Promise<Session> {
  const response = await fetch(resolveServiceURL(`sessions/thread/${threadId}`));
  if (!response.ok) {
    throw new Error(`Failed to get session by thread: ${response.statusText}`);
  }
  return response.json();
}

export async function archiveSession(
  sessionId: string
): Promise<{ status: string; session_id: string }> {
  const response = await fetch(
    resolveServiceURL(`sessions/${sessionId}/archive`),
    { method: "POST" }
  );
  if (!response.ok) {
    throw new Error(`Failed to archive session: ${response.statusText}`);
  }
  return response.json();
}

export async function restoreSession(
  sessionId: string
): Promise<{ status: string; session_id: string }> {
  const response = await fetch(
    resolveServiceURL(`sessions/${sessionId}/restore`),
    { method: "POST" }
  );
  if (!response.ok) {
    throw new Error(`Failed to restore session: ${response.statusText}`);
  }
  return response.json();
}

export async function exportSession(
  sessionId: string
): Promise<{ session: Session; memory: Record<string, unknown> | null }> {
  const response = await fetch(
    resolveServiceURL(`sessions/${sessionId}/export`)
  );
  if (!response.ok) {
    throw new Error(`Failed to export session: ${response.statusText}`);
  }
  return response.json();
}
