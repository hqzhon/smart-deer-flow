// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { resolveServiceURL } from "./resolve-service-url";

export interface SandboxInfo {
  id: string;
  status: string;
  created_at: string;
  image: string;
  cpu_limit: number;
  memory_limit: string;
  network_enabled: boolean;
  work_dir: string;
}

export interface SandboxMetrics {
  cpu_usage: number;
  memory_usage: number;
  memory_used: string;
  memory_total: string;
  disk_usage: number;
  disk_used: string;
  disk_total: string;
  network_rx: string;
  network_tx: string;
  uptime: number;
  process_count: number;
}

export interface SandboxStatus {
  info: SandboxInfo;
  metrics?: SandboxMetrics;
  last_activity?: string;
  logs: string[];
}

export interface SandboxCreateRequest {
  image?: string;
  memory_limit?: string;
  cpu_limit?: number;
  network_enabled?: boolean;
  timeout?: number;
}

export interface CommandRequest {
  command: string;
  timeout?: number;
}

export interface CommandResponse {
  output: string;
  exit_code: number;
  duration: number;
}

export interface FileInfo {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  modified?: string;
  permissions?: string;
}

export interface FileContent {
  path: string;
  content: string;
  encoding: string;
}

export async function listSandboxes(): Promise<{
  sandboxes: Array<SandboxInfo & { status: string }>;
  total: number;
  manager_stats: Record<string, unknown>;
}> {
  const response = await fetch(resolveServiceURL("sandbox"));
  if (!response.ok) {
    throw new Error(`Failed to list sandboxes: ${response.statusText}`);
  }
  return response.json();
}

export async function createSandbox(
  request: SandboxCreateRequest
): Promise<SandboxInfo> {
  const response = await fetch(resolveServiceURL("sandbox"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(`Failed to create sandbox: ${response.statusText}`);
  }
  return response.json();
}

export async function getSandboxStatus(
  sandboxId: string
): Promise<SandboxStatus> {
  const response = await fetch(resolveServiceURL(`sandbox/${sandboxId}`));
  if (!response.ok) {
    throw new Error(`Failed to get sandbox status: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteSandbox(
  sandboxId: string
): Promise<{ status: string; sandbox_id: string }> {
  const response = await fetch(resolveServiceURL(`sandbox/${sandboxId}`), {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Failed to delete sandbox: ${response.statusText}`);
  }
  return response.json();
}

export async function executeCommand(
  sandboxId: string,
  request: CommandRequest
): Promise<CommandResponse> {
  const response = await fetch(
    resolveServiceURL(`sandbox/${sandboxId}/execute`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }
  );
  if (!response.ok) {
    throw new Error(`Failed to execute command: ${response.statusText}`);
  }
  return response.json();
}

export async function listFiles(
  sandboxId: string,
  path = "/workspace"
): Promise<FileInfo[]> {
  const response = await fetch(
    `${resolveServiceURL(`sandbox/${sandboxId}/files`)}?path=${encodeURIComponent(path)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to list files: ${response.statusText}`);
  }
  return response.json();
}

export async function readFile(
  sandboxId: string,
  filePath: string
): Promise<FileContent> {
  const response = await fetch(
    resolveServiceURL(`sandbox/${sandboxId}/files/${encodeURIComponent(filePath)}`)
  );
  if (!response.ok) {
    throw new Error(`Failed to read file: ${response.statusText}`);
  }
  return response.json();
}

export async function getSandboxMetrics(
  sandboxId: string
): Promise<SandboxMetrics> {
  const response = await fetch(
    resolveServiceURL(`sandbox/${sandboxId}/metrics`)
  );
  if (!response.ok) {
    throw new Error(`Failed to get sandbox metrics: ${response.statusText}`);
  }
  return response.json();
}

export function createSandboxTerminalStream(
  sandboxId: string
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}/api/sandbox/${sandboxId}/terminal`);
}
