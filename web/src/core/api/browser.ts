// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { resolveServiceURL } from "./resolve-service-url";

export interface BrowserAction {
  action: string;
  url?: string;
  index?: number;
  text?: string;
  scroll_amount?: number;
  tab_id?: number;
  goal?: string;
  keys?: string;
  seconds?: number;
  selector?: string;
}

export interface BrowserActionRecord {
  timestamp: string;
  action: string;
  params: Record<string, unknown>;
  result: string;
  success: boolean;
}

export interface BrowserState {
  session_id: string;
  url: string;
  title: string;
  tabs: Array<{ id: number; url: string; title: string }>;
  interactive_elements: Array<{ index: number; tag: string; text: string }>;
  last_screenshot?: string;
  actions: BrowserActionRecord[];
  created_at: string;
  updated_at: string;
}

export interface BrowserActionResult {
  output?: string;
  error?: string;
  base64_image?: string;
}

export interface BrowserSessionInfo {
  session_id: string;
  url: string;
  title: string;
  tabs_count: number;
  actions_count: number;
  created_at: string;
  updated_at: string;
}

export async function executeBrowserAction(
  sessionId: string,
  action: BrowserAction
): Promise<{ success: boolean; session_id: string; result: BrowserActionResult }> {
  const response = await fetch(
    resolveServiceURL(`browser/${sessionId}/execute`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }
  );
  if (!response.ok) {
    throw new Error(`Failed to execute browser action: ${response.statusText}`);
  }
  return response.json();
}

export async function getBrowserScreenshot(
  sessionId: string
): Promise<{ success: boolean; session_id: string; base64_image?: string; url?: string; error?: string }> {
  const response = await fetch(
    resolveServiceURL(`browser/${sessionId}/screenshot`)
  );
  if (!response.ok) {
    throw new Error(`Failed to get screenshot: ${response.statusText}`);
  }
  return response.json();
}

export async function getBrowserState(
  sessionId: string
): Promise<BrowserState> {
  const response = await fetch(
    resolveServiceURL(`browser/${sessionId}/state`)
  );
  if (!response.ok) {
    throw new Error(`Failed to get browser state: ${response.statusText}`);
  }
  return response.json();
}

export async function getBrowserActions(
  sessionId: string,
  limit = 50
): Promise<{ session_id: string; actions: BrowserActionRecord[]; total: number }> {
  const response = await fetch(
    `${resolveServiceURL(`browser/${sessionId}/actions`)}?limit=${limit}`
  );
  if (!response.ok) {
    throw new Error(`Failed to get browser actions: ${response.statusText}`);
  }
  return response.json();
}

export async function closeBrowserSession(
  sessionId: string
): Promise<{ success: boolean; message: string }> {
  const response = await fetch(
    resolveServiceURL(`browser/${sessionId}`),
    { method: "DELETE" }
  );
  if (!response.ok) {
    throw new Error(`Failed to close browser session: ${response.statusText}`);
  }
  return response.json();
}

export async function listBrowserSessions(): Promise<{
  sessions: BrowserSessionInfo[];
  total: number;
}> {
  const response = await fetch(resolveServiceURL("browser"));
  if (!response.ok) {
    throw new Error(`Failed to list browser sessions: ${response.statusText}`);
  }
  return response.json();
}

export function createBrowserStream(
  sessionId: string
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}/api/browser/${sessionId}/stream`);
}
