"use client";
import { useEffect } from "react";
import type { ReactNode } from "react";
import { loadConfig } from "~/core/api/config";

interface ConfigProviderProps {
  children: ReactNode;
}

export default function ConfigProvider({ children }: ConfigProviderProps) {

  useEffect(() => {
    async function fetchConfig() {
      const config = await loadConfig();
      window.__deerflowConfig = config;
    }
    fetchConfig();
  }, []);

  return <>{children}</>;
}