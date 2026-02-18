"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Box,
  Cpu,
  HardDrive,
  MemoryStick,
  Network,
  Play,
  Square,
  Terminal,
  RefreshCw,
  Trash2,
  FolderOpen,
  Loader2,
  AlertCircle,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { Progress } from "~/components/ui/progress";
import { ScrollArea } from "~/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";

import {
  getSandboxStatus,
  executeCommand,
  listFiles,
  deleteSandbox,
  type SandboxStatus as SandboxStatusType,
  type FileInfo,
} from "~/core/api/sandbox";

interface SandboxStatusProps {
  sandboxId: string;
  onDelete?: () => void;
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function SandboxStatus({
  sandboxId,
  onDelete,
  className = "",
  autoRefresh = true,
  refreshInterval = 5000,
}: SandboxStatusProps) {
  const [status, setStatus] = useState<SandboxStatusType | null>(null);
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [commandInput, setCommandInput] = useState("");
  const [commandOutput, setCommandOutput] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState("/workspace");

  const fetchStatus = useCallback(async () => {
    if (!sandboxId) return;

    try {
      const data = await getSandboxStatus(sandboxId);
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [sandboxId]);

  const fetchFiles = useCallback(async (path: string) => {
    if (!sandboxId) return;

    try {
      const data = await listFiles(sandboxId, path);
      setFiles(data);
      setCurrentPath(path);
    } catch (err) {
      console.error("Failed to fetch files:", err);
    }
  }, [sandboxId]);

  useEffect(() => {
    fetchStatus();
    fetchFiles(currentPath);
  }, [sandboxId, fetchStatus, fetchFiles, currentPath]);

  useEffect(() => {
    if (autoRefresh && sandboxId) {
      const interval = setInterval(fetchStatus, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, sandboxId, refreshInterval, fetchStatus]);

  const handleExecuteCommand = async () => {
    if (!commandInput.trim() || !sandboxId) return;

    setIsLoading(true);
    try {
      const result = await executeCommand(sandboxId, {
        command: commandInput,
      });
      setCommandOutput(
        `$ ${commandInput}\n${result.output}\n[Exit: ${result.exit_code}, Time: ${result.duration.toFixed(2)}s]`
      );
      setCommandInput("");
      fetchStatus();
    } catch (err) {
      setCommandOutput(`Error: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!sandboxId) return;

    try {
      await deleteSandbox(sandboxId);
      onDelete?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete sandbox");
    }
  };

  const handleNavigate = (file: FileInfo) => {
    if (file.is_dir) {
      fetchFiles(file.path);
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const metrics = status?.metrics;

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Box className="h-5 w-5" />
            Sandbox Status
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge
              variant={status?.info?.status === "running" ? "default" : "secondary"}
            >
              {status?.info?.status || "unknown"}
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              onClick={fetchStatus}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDelete}
              className="text-destructive"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            {error}
          </div>
        )}

        {status?.info && (
          <div className="mb-4 rounded-md bg-muted p-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-mono">{status.info.id.slice(0, 8)}</span>
              <span className="text-muted-foreground">|</span>
              <span>{status.info.image}</span>
            </div>
          </div>
        )}

        <Tabs defaultValue="metrics" className="w-full">
          <TabsList className="mb-2">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="terminal">Terminal</TabsTrigger>
            <TabsTrigger value="files">Files</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-muted-foreground" />
                    CPU
                  </div>
                  <span className="font-mono">
                    {metrics?.cpu_usage?.toFixed(1) || 0}%
                  </span>
                </div>
                <Progress value={metrics?.cpu_usage || 0} />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <MemoryStick className="h-4 w-4 text-muted-foreground" />
                    Memory
                  </div>
                  <span className="font-mono">
                    {metrics?.memory_usage?.toFixed(1) || 0}%
                  </span>
                </div>
                <Progress value={metrics?.memory_usage || 0} />
                <div className="text-xs text-muted-foreground text-right">
                  {metrics?.memory_used || "0B"} / {metrics?.memory_total || "0B"}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-muted-foreground" />
                    Disk
                  </div>
                  <span className="font-mono">
                    {metrics?.disk_usage?.toFixed(1) || 0}%
                  </span>
                </div>
                <Progress value={metrics?.disk_usage || 0} />
                <div className="text-xs text-muted-foreground text-right">
                  {metrics?.disk_used || "0B"} / {metrics?.disk_total || "0B"}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Network className="h-4 w-4 text-muted-foreground" />
                    Network
                  </div>
                  <span className="font-mono text-xs">
                    ↓{metrics?.network_rx || "0B"} ↑{metrics?.network_tx || "0B"}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between text-sm text-muted-foreground pt-2 border-t">
              <div className="flex items-center gap-4">
                <span>Uptime: {formatUptime(metrics?.uptime || 0)}</span>
                <span>Processes: {metrics?.process_count || 0}</span>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="terminal" className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Enter command..."
                value={commandInput}
                onChange={(e) => setCommandInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleExecuteCommand()}
                className="font-mono"
              />
              <Button onClick={handleExecuteCommand} disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
            </div>

            <ScrollArea className="h-[200px] rounded-md border bg-muted p-2">
              <pre className="font-mono text-xs whitespace-pre-wrap">
                {commandOutput || "No commands executed yet."}
              </pre>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="files">
            <div className="mb-2 text-sm text-muted-foreground">
              <FolderOpen className="inline h-4 w-4 mr-1" />
              {currentPath}
            </div>

            <ScrollArea className="h-[200px] rounded-md border">
              {files.length > 0 ? (
                <div className="divide-y">
                  {files.map((file) => (
                    <div
                      key={file.path}
                      className={`flex items-center justify-between p-2 text-sm ${
                        file.is_dir ? "hover:bg-muted cursor-pointer" : ""
                      }`}
                      onClick={() => file.is_dir && handleNavigate(file)}
                    >
                      <div className="flex items-center gap-2">
                        <span className={file.is_dir ? "text-primary" : ""}>
                          {file.is_dir ? "📁" : "📄"} {file.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        {!file.is_dir && <span>{file.size}B</span>}
                        {file.permissions && (
                          <span className="font-mono">{file.permissions}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No files found
                </div>
              )}
            </ScrollArea>
          </TabsContent>

          <TabsContent value="logs">
            <ScrollArea className="h-[200px] rounded-md border bg-muted p-2">
              {status?.logs && status.logs.length > 0 ? (
                <div className="font-mono text-xs space-y-1">
                  {status.logs.map((log, i) => (
                    <div key={i} className="whitespace-pre-wrap">
                      {log}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No logs available
                </div>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
