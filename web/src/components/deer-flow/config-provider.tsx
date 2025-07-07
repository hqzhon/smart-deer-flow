"use client";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { loadConfig } from "~/core/api/config";

interface ConfigProviderProps {
  children: ReactNode;
}

export default function ConfigProvider({ children }: ConfigProviderProps) {
  const [isConfigLoaded, setIsConfigLoaded] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchConfig() {
      try {
        const config = await loadConfig();
        window.__deerflowConfig = config;
        setIsConfigLoaded(true);
      } catch (error) {
        console.error("Failed to load config:", error);
        setConfigError(error instanceof Error ? error.message : "Unknown error");
      }
    }
    fetchConfig();
  }, []);

  // Show loading state while config is being fetched
  if (!isConfigLoaded && !configError) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="text-center">
          <div className="mb-4 h-8 w-8 animate-spin rounded-full border-4 border-gray-300 border-t-blue-600 mx-auto"></div>
          <p className="text-gray-600">Loading configuration...</p>
        </div>
      </div>
    );
  }

  // Show error state if config loading failed
  if (configError) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="text-center">
          <div className="mb-4 text-red-500">
            <svg className="mx-auto h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h2 className="mb-2 text-xl font-semibold text-gray-900">Configuration Error</h2>
          <p className="text-gray-600 mb-4">Failed to load application configuration: {configError}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Render children only after config is successfully loaded
  return <>{children}</>;
}