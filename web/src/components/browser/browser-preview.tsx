"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Monitor,
  RefreshCw,
  ExternalLink,
  Camera,
  Clock,
  Globe,
  AlertCircle,
  Play,
  Square,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { ScrollArea } from "~/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";

import {
  getBrowserState,
  getBrowserScreenshot,
  executeBrowserAction,
  createBrowserStream,
  type BrowserState,
  type BrowserActionRecord,
} from "~/core/api/browser";

interface BrowserPreviewProps {
  sessionId: string;
  onAction?: (action: string, params: Record<string, unknown>) => void;
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function BrowserPreview({
  sessionId,
  onAction,
  className = "",
  autoRefresh = false,
  refreshInterval = 2000,
}: BrowserPreviewProps) {
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [state, setState] = useState<BrowserState | null>(null);
  const [actions, setActions] = useState<BrowserActionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const fetchState = useCallback(async () => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await getBrowserState(sessionId);
      setState(data);
      setActions(data.actions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const fetchScreenshot = useCallback(async () => {
    if (!sessionId) return;

    try {
      const data = await getBrowserScreenshot(sessionId);
      if (data.success && data.base64_image) {
        setScreenshot(`data:image/jpeg;base64,${data.base64_image}`);
      }
    } catch (err) {
      console.error("Failed to fetch screenshot:", err);
    }
  }, [sessionId]);

  const startStreaming = useCallback(() => {
    if (!sessionId || wsRef.current) return;

    const ws = createBrowserStream(sessionId);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsStreaming(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "screenshot" && data.base64_image) {
          setScreenshot(`data:image/jpeg;base64,${data.base64_image}`);
          if (data.url) {
            setState((prev) =>
              prev
                ? { ...prev, url: data.url, title: data.title || prev.title }
                : null
            );
          }
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      setIsStreaming(false);
    };

    ws.onclose = () => {
      setIsStreaming(false);
      wsRef.current = null;
    };
  }, [sessionId]);

  const stopStreaming = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  useEffect(() => {
    if (sessionId) {
      fetchState();
      fetchScreenshot();
    }
  }, [sessionId, fetchState, fetchScreenshot]);

  useEffect(() => {
    if (autoRefresh && sessionId && !isStreaming) {
      const interval = setInterval(fetchScreenshot, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, sessionId, refreshInterval, fetchScreenshot, isStreaming]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleRefresh = () => {
    fetchState();
    fetchScreenshot();
  };

  const handleTakeScreenshot = async () => {
    if (!sessionId) return;
    await fetchScreenshot();
  };

  const handleExecuteAction = async (action: string, params: Record<string, unknown> = {}) => {
    if (!sessionId) return;

    try {
      const result = await executeBrowserAction(sessionId, { action, ...params });
      if (result.result?.base64_image) {
        setScreenshot(`data:image/jpeg;base64,${result.result.base64_image}`);
      }
      onAction?.(action, params);
      fetchState();
    } catch (err) {
      console.error("Failed to execute action:", err);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString();
  };

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Monitor className="h-5 w-5" />
            Browser Preview
          </CardTitle>
          <div className="flex items-center gap-2">
            {state && (
              <Badge variant="outline" className="text-xs">
                <Globe className="mr-1 h-3 w-3" />
                {state.tabs?.length || 0} tabs
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={isStreaming ? stopStreaming : startStreaming}
              title={isStreaming ? "Stop streaming" : "Start streaming"}
            >
              {isStreaming ? (
                <Square className="h-4 w-4 text-destructive" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
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

        <Tabs defaultValue="preview" className="w-full">
          <TabsList className="mb-2">
            <TabsTrigger value="preview">Preview</TabsTrigger>
            <TabsTrigger value="actions">Actions</TabsTrigger>
            <TabsTrigger value="elements">Elements</TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="space-y-4">
            {state && (
              <div className="rounded-md bg-muted p-2">
                <div className="flex items-center gap-2 text-sm">
                  <Globe className="h-4 w-4 text-muted-foreground" />
                  <span className="truncate font-medium">{state.title}</span>
                </div>
                <div className="mt-1 truncate text-xs text-muted-foreground">
                  {state.url}
                </div>
              </div>
            )}

            <div className="relative aspect-video overflow-hidden rounded-md border bg-muted">
              {screenshot ? (
                <img
                  src={screenshot}
                  alt="Browser screenshot"
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <Monitor className="mx-auto h-12 w-12 opacity-50" />
                    <p className="mt-2 text-sm">No preview available</p>
                  </div>
                </div>
              )}

              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                  <RefreshCw className="h-8 w-8 animate-spin text-primary" />
                </div>
              )}

              {isStreaming && (
                <div className="absolute top-2 right-2">
                  <Badge variant="default" className="animate-pulse">
                    Live
                  </Badge>
                </div>
              )}
            </div>

            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleTakeScreenshot}
                disabled={!sessionId}
              >
                <Camera className="mr-2 h-4 w-4" />
                Screenshot
              </Button>
              {state?.url && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.open(state.url, "_blank")}
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open
                </Button>
              )}
            </div>
          </TabsContent>

          <TabsContent value="actions">
            <ScrollArea className="h-[300px]">
              {actions.length > 0 ? (
                <div className="space-y-2">
                  {actions.map((action, index) => (
                    <div
                      key={`${action.timestamp}-${index}`}
                      className="flex items-start gap-2 rounded-md border p-2"
                    >
                      <Clock className="mt-0.5 h-4 w-4 text-muted-foreground" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{action.action}</span>
                          {!action.success && (
                            <Badge variant="destructive" className="text-xs">
                              Error
                            </Badge>
                          )}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {formatDate(action.timestamp)}
                        </div>
                        {action.result && (
                          <div className="mt-1 text-sm">{action.result}</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No actions recorded
                </div>
              )}
            </ScrollArea>
          </TabsContent>

          <TabsContent value="elements">
            <ScrollArea className="h-[300px]">
              {state?.interactive_elements && state.interactive_elements.length > 0 ? (
                <div className="space-y-1">
                  {state.interactive_elements.map((element) => (
                    <div
                      key={element.index}
                      className="flex items-center gap-2 rounded-md p-2 hover:bg-muted cursor-pointer"
                      onClick={() =>
                        handleExecuteAction("click_element", { index: element.index })
                      }
                    >
                      <Badge variant="outline" className="font-mono text-xs">
                        [{element.index}]
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {element.tag}
                      </span>
                      <span className="truncate text-sm">{element.text}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No interactive elements found
                </div>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
