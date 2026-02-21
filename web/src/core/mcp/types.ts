export interface MCPServerConfig extends Record<string, unknown> {
  enabled: boolean;
  description: string;
}

export interface MCPConfig {
  mcp_servers: Record<string, MCPServerConfig>;
}

export interface MCPToolMetadata {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export interface MCPServerMetadata {
  name: string;
  description?: string;
  tools: MCPToolMetadata[];
  enabled?: boolean;
  transport?: "stdio" | "sse";
  command?: string;
  args?: string[];
  url?: string;
  env?: Record<string, string>;
  createdAt?: string;
  updatedAt?: string;
}

export type SimpleStdioMCPServerMetadata = {
  command: string;
  args?: string[];
  env?: Record<string, string>;
};

export type SimpleSSEMCPServerMetadata = {
  url: string;
  env?: Record<string, string>;
};

export type SimpleMCPServerMetadata =
  | SimpleStdioMCPServerMetadata
  | SimpleSSEMCPServerMetadata;

export function findMCPTool(
  servers: MCPServerMetadata[],
  toolName: string,
): { server: MCPServerMetadata; tool: MCPToolMetadata } | undefined {
  for (const server of servers) {
    const tool = server.tools.find((t) => t.name === toolName);
    if (tool) {
      return { server, tool };
    }
  }
  return undefined;
}
