// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import { Calendar, MessageSquare, MoreVertical, Trash2, Archive, RotateCcw } from "lucide-react";
import { useState, useEffect, useCallback } from "react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { Input } from "~/components/ui/input";
import { ScrollArea } from "~/components/ui/scroll-area";

import {
  listSessions,
  deleteSession,
  archiveSession,
  restoreSession,
  type Session,
} from "~/core/api/session";

interface SessionListProps {
  onSelectSession?: (session: Session) => void;
  onCreateSession?: () => void;
  refreshTrigger?: number;
}

export function SessionList({
  onSelectSession,
  onCreateSession,
  refreshTrigger,
}: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);

  const fetchSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await listSessions({
        page,
        page_size: 20,
        search: searchQuery || undefined,
        status: "active",
      });
      setSessions((prev) =>
        page === 1 ? response.sessions : [...prev, ...response.sessions]
      );
      setHasMore(response.has_more);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    } finally {
      setLoading(false);
    }
  }, [page, searchQuery]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions, refreshTrigger]);

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const handleArchive = async (sessionId: string) => {
    try {
      await archiveSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    } catch (err) {
      console.error("Failed to archive session:", err);
    }
  };

  const handleRestore = async (sessionId: string) => {
    try {
      await restoreSession(sessionId);
      fetchSessions();
    } catch (err) {
      console.error("Failed to restore session:", err);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b">
        <div className="flex items-center gap-2">
          <Input
            placeholder="Search sessions..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setPage(1);
            }}
            className="flex-1"
          />
          {onCreateSession && (
            <Button onClick={onCreateSession} size="sm">
              New
            </Button>
          )}
        </div>
      </div>

      <ScrollArea className="flex-1">
        {loading && sessions.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-muted-foreground">
            Loading sessions...
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-32 text-destructive">
            {error}
          </div>
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-muted-foreground">
            <MessageSquare className="h-8 w-8 mb-2" />
            <p>No sessions found</p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {sessions.map((session) => (
              <Card
                key={session.id}
                className="cursor-pointer hover:bg-accent/50 transition-colors"
                onClick={() => onSelectSession?.(session)}
              >
                <CardHeader className="p-3 pb-2">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-sm font-medium line-clamp-1">
                      {session.title || session.research_topic || "Untitled Session"}
                    </CardTitle>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                        <Button variant="ghost" size="icon" className="h-6 w-6">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {session.status === "archived" ? (
                          <DropdownMenuItem onClick={() => handleRestore(session.id)}>
                            <RotateCcw className="h-4 w-4 mr-2" />
                            Restore
                          </DropdownMenuItem>
                        ) : (
                          <DropdownMenuItem onClick={() => handleArchive(session.id)}>
                            <Archive className="h-4 w-4 mr-2" />
                            Archive
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuItem
                          onClick={() => handleDelete(session.id)}
                          className="text-destructive"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </CardHeader>
                <CardContent className="p-3 pt-0">
                  {session.last_message_preview && (
                    <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                      {session.last_message_preview}
                    </p>
                  )}
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {formatDate(session.updated_at)}
                    </div>
                    <div className="flex items-center gap-2">
                      {session.tags.slice(0, 2).map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs px-1">
                          {tag}
                        </Badge>
                      ))}
                      <Badge variant="outline" className="text-xs">
                        {session.message_count} msgs
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}

            {hasMore && (
              <div className="flex justify-center py-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => p + 1)}
                  disabled={loading}
                >
                  {loading ? "Loading..." : "Load More"}
                </Button>
              </div>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
