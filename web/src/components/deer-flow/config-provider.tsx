"use client";
import { useEffect, useState } from "react";
import { loadConfig } from "~/core/api/config";

export default function ConfigProvider({ children }) {

  useEffect(() => {
    async function fetchConfig() {
      const config = await loadConfig();
      window.__deerflowConfig = config;
    }
    fetchConfig();
  }, []);

  return <>{children}</>;
}