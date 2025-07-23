// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import { cn } from "~/lib/utils";

interface MermaidProps {
  chart: string;
  className?: string;
  id?: string;
  isStreaming?: boolean;
}

export function Mermaid({ chart, className, id, isStreaming = false }: MermaidProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isInitialized, setIsInitialized] = useState(false);
  const chartId = useRef(id || `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    // Initialize mermaid only once
    if (!isInitialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: "default",
        securityLevel: "loose",
        fontFamily: "inherit",
        fontSize: 14,
        flowchart: {
          useMaxWidth: true,
          htmlLabels: true,
        },
        sequence: {
          useMaxWidth: true,
        },
        gantt: {
          useMaxWidth: true,
        },
      });
      setIsInitialized(true);
    }
  }, [isInitialized]);

  useEffect(() => {
    const renderChart = async () => {
      // Don't render during streaming to avoid syntax errors from incomplete charts
      if (!isInitialized || !chart.trim() || isStreaming) {
        if (isStreaming) {
          // Clear any previous error when streaming starts
          setError("");
          setSvg("");
        }
        return;
      }
      
      try {
        setError("");
        setSvg("");
        
        // Clean the chart content
        const cleanChart = chart.trim();
        
        // Generate a unique ID for each render
        const uniqueId = `${chartId.current}-${Date.now()}`;
        
        // Validate and render the chart
        const { svg: renderedSvg } = await mermaid.render(uniqueId, cleanChart);
        setSvg(renderedSvg);
      } catch (err) {
        console.error("Mermaid rendering error:", err);
        setError(err instanceof Error ? err.message : "Failed to render chart");
      }
    };

    renderChart();
  }, [chart, isInitialized, isStreaming]);

  if (error) {
    return (
      <div className={cn("rounded border border-red-200 bg-red-50 p-4", className)}>
        <div className="text-sm font-medium text-red-800">Mermaid Error</div>
        <div className="mt-1 text-sm text-red-700">{error}</div>
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-red-600 hover:text-red-800">
            Show chart source
          </summary>
          <pre className="mt-2 whitespace-pre-wrap text-xs text-red-600">{chart}</pre>
        </details>
      </div>
    );
  }

  // Show loading state during streaming
  if (isStreaming || (!svg && chart.trim())) {
    return (
      <div className={cn(
        "mermaid-diagram flex items-center justify-center rounded border bg-white p-8 dark:bg-gray-900",
        className
      )}>
        <div className="flex items-center space-x-2 text-muted-foreground">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          <span className="text-sm">Preparing diagram...</span>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className={cn(
        "mermaid-diagram flex justify-center overflow-x-auto rounded border bg-white p-4 dark:bg-gray-900",
        className
      )}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

export default Mermaid;