"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Activity,
  Cpu,
  Database,
  DollarSign,
  Gauge,
  Loader2,
  MemoryStick,
  RefreshCw,
  Server,
  Timer,
  Zap,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import { ScrollArea } from "~/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";

import {
  getMetricsDashboard,
  getMetricsHistory,
  type MetricsDashboard as MetricsDashboardType,
  type MetricsHistory,
} from "~/core/api/metrics";

interface MetricsDashboardProps {
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function MetricsDashboard({
  className = "",
  autoRefresh = true,
  refreshInterval = 5000,
}: MetricsDashboardProps) {
  const [dashboard, setDashboard] = useState<MetricsDashboardType | null>(null);
  const [history, setHistory] = useState<MetricsHistory | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboard = useCallback(async () => {
    setIsLoading(true);
    try {
      const [dashboardData, historyData] = await Promise.all([
        getMetricsDashboard(),
        getMetricsHistory(3600),
      ]);
      setDashboard(dashboardData);
      setHistory(historyData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboard();
  }, [fetchDashboard]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchDashboard, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, fetchDashboard]);

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const formatBytes = (str: string) => str;

  const system = dashboard?.system;
  const llm = dashboard?.llm;
  const context = dashboard?.context;
  const performance = dashboard?.performance;

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Performance Dashboard
        </h2>
        <div className="flex items-center gap-2">
          {dashboard && (
            <Badge variant="outline">
              Uptime: {formatUptime(dashboard.uptime)}
            </Badge>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={fetchDashboard}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              CPU Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {system?.cpu_usage?.toFixed(1) || 0}%
            </div>
            <Progress value={system?.cpu_usage || 0} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <MemoryStick className="h-4 w-4 text-muted-foreground" />
              Memory
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {system?.memory_usage?.toFixed(1) || 0}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {formatBytes(system?.memory_used || "0B")} / {formatBytes(system?.memory_total || "0B")}
            </div>
            <Progress value={system?.memory_usage || 0} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <DollarSign className="h-4 w-4 text-muted-foreground" />
              LLM Cost
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${llm?.total_cost?.toFixed(4) || "0.00"}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {llm?.total_tokens?.toLocaleString() || 0} tokens
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Timer className="h-4 w-4 text-muted-foreground" />
              Avg Response
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {performance?.average_response_time?.toFixed(0) || 0}ms
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              P95: {performance?.p95_response_time?.toFixed(0) || 0}ms
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="llm" className="w-full">
        <TabsList>
          <TabsTrigger value="llm">LLM Metrics</TabsTrigger>
          <TabsTrigger value="context">Context</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>

        <TabsContent value="llm" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Total Requests</div>
                <div className="text-xl font-bold">{llm?.total_requests || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Input Tokens</div>
                <div className="text-xl font-bold">{llm?.input_tokens?.toLocaleString() || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Output Tokens</div>
                <div className="text-xl font-bold">{llm?.output_tokens?.toLocaleString() || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Error Rate</div>
                <div className="text-xl font-bold">{llm?.error_rate?.toFixed(2) || 0}%</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground mb-2">Requests / Minute</div>
                <div className="text-2xl font-bold">{llm?.requests_per_minute?.toFixed(2) || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground mb-2">Tokens / Minute</div>
                <div className="text-2xl font-bold">{llm?.tokens_per_minute?.toFixed(0) || 0}</div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="context" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Total Sessions</div>
                <div className="text-xl font-bold">{context?.total_sessions || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Active Sessions</div>
                <div className="text-xl font-bold">{context?.active_sessions || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Total Messages</div>
                <div className="text-xl font-bold">{context?.total_messages?.toLocaleString() || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Cache Hit Rate</div>
                <div className="text-xl font-bold">{context?.cache_hit_rate?.toFixed(1) || 0}%</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground mb-2">Avg Context Size</div>
                <div className="text-2xl font-bold">
                  {context?.average_context_size?.toLocaleString() || 0} tokens
                </div>
                <Progress
                  value={Math.min((context?.average_context_size || 0) / 128000 * 100, 100)}
                  className="mt-2"
                />
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground mb-2">Compression Ratio</div>
                <div className="text-2xl font-bold">
                  {context?.compression_ratio?.toFixed(2) || 0}x
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Active Connections</div>
                <div className="text-xl font-bold">
                  {performance?.active_connections || 0} / {performance?.max_connections || 0}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">Connection Utilization</div>
                <div className="text-xl font-bold">
                  {performance?.connection_utilization?.toFixed(1) || 0}%
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">P50 Response</div>
                <div className="text-xl font-bold">
                  {performance?.p50_response_time?.toFixed(0) || 0}ms
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="text-sm text-muted-foreground">P99 Response</div>
                <div className="text-xl font-bold">
                  {performance?.p99_response_time?.toFixed(0) || 0}ms
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Model Usage Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              {llm?.models_used && Object.keys(llm.models_used).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(llm.models_used).map(([model, count]) => (
                    <div
                      key={model}
                      className="flex items-center justify-between p-2 rounded-md bg-muted"
                    >
                      <span className="font-mono text-sm">{model}</span>
                      <Badge variant="secondary">{count.toLocaleString()} requests</Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-4">
                  No model usage data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
