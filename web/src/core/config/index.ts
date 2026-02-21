import { env } from "@/env";

export * from "./types";
export * from "./agents";

export function getBackendBaseURL() {
  if (env.NEXT_PUBLIC_BACKEND_BASE_URL) {
    return env.NEXT_PUBLIC_BACKEND_BASE_URL;
  }
  return "";
}

export function getLangGraphBaseURL() {
  if (env.NEXT_PUBLIC_LANGGRAPH_BASE_URL) {
    return env.NEXT_PUBLIC_LANGGRAPH_BASE_URL;
  }
  if (typeof window !== "undefined") {
    return `${window.location.origin}/api/langgraph`;
  }
  return "http://localhost:2026/api/langgraph";
}
