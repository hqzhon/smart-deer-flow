import type {
  SimpleMCPServerMetadata,
  SimpleSSEMCPServerMetadata,
  SimpleStdioMCPServerMetadata,
} from "./types";

export function isStdioMCPServer(
  server: SimpleMCPServerMetadata,
): server is SimpleStdioMCPServerMetadata {
  return "command" in server;
}

export function isSSEMCPServer(
  server: SimpleMCPServerMetadata,
): server is SimpleSSEMCPServerMetadata {
  return "url" in server;
}
