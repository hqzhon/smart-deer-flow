// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { motion } from "framer-motion";
import {
  BarChart3,
  Brain,
  CheckCircle,
  Clock,
  FileText,
  HelpCircle,
  Lightbulb,
  TrendingUp,
} from "lucide-react";
import { useMemo } from "react";

import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Separator } from "~/components/ui/separator";
import type { Message } from "~/core/messages";
import { cn } from "~/lib/utils";

export function StructuredDataBlock({
  className,
  message,
}: {
  className?: string;
  message: Message;
}) {
  const hasStructuredData = useMemo(() => {
    return (
      !!message.reflectionInsights ||
      !!message.isolationMetrics ||
      !!message.researchProgress ||
      (message.nodeUpdates && message.nodeUpdates.length > 0)
    );
  }, [message]);

  if (!hasStructuredData) {
    return null;
  }

  return (
    <motion.div
      className={cn("mt-4 space-y-4", className)}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {message.reflectionInsights && (
        <ReflectionInsightsCard insights={message.reflectionInsights} />
      )}
      {message.isolationMetrics && (
        <IsolationMetricsCard metrics={message.isolationMetrics} />
      )}
      {message.researchProgress && (
        <ResearchProgressCard progress={message.researchProgress} />
      )}
      {message.nodeUpdates && message.nodeUpdates.length > 0 && (
        <NodeUpdatesCard updates={message.nodeUpdates} />
      )}
    </motion.div>
  );
}

function ReflectionInsightsCard({
  insights,
}: {
  insights: NonNullable<Message["reflectionInsights"]>;
}) {
  return (
    <Card className="border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/20">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
          <Brain size={18} />
          Reflection Insights
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {insights.comprehensive_report && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-medium">
              <FileText size={14} />
              Comprehensive Report
            </div>
            <p className="text-sm text-muted-foreground">
              {insights.comprehensive_report}
            </p>
          </div>
        )}
        
        {insights.knowledge_gaps && insights.knowledge_gaps.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-medium">
              <HelpCircle size={14} />
              Knowledge Gaps
            </div>
            <div className="flex flex-wrap gap-1">
              {insights.knowledge_gaps.map((gap, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {gap}
                </Badge>
              ))}
            </div>
          </div>
        )}
        
        {insights.follow_up_queries && insights.follow_up_queries.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-medium">
              <Lightbulb size={14} />
              Follow-up Queries
            </div>
            <ul className="space-y-1">
              {insights.follow_up_queries.map((query, index) => (
                <li key={index} className="text-sm text-muted-foreground">
                  • {query}
                </li>
              ))}
            </ul>
          </div>
        )}
        
        <div className="flex items-center justify-between pt-2">
          {insights.is_sufficient !== undefined && (
            <div className="flex items-center gap-2">
              <CheckCircle
                size={14}
                className={cn(
                  insights.is_sufficient
                    ? "text-green-600 dark:text-green-400"
                    : "text-yellow-600 dark:text-yellow-400"
                )}
              />
              <span className="text-sm">
                {insights.is_sufficient ? "Sufficient" : "Needs More Info"}
              </span>
            </div>
          )}
          
          {insights.confidence_score !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Confidence:</span>
              <Badge variant="secondary">
                {Math.round(insights.confidence_score * 100)}%
              </Badge>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function IsolationMetricsCard({
  metrics,
}: {
  metrics: NonNullable<Message["isolationMetrics"]>;
}) {
  return (
    <Card className="border-green-200 bg-green-50/50 dark:border-green-800 dark:bg-green-950/20">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-green-700 dark:text-green-300">
          <BarChart3 size={18} />
          Isolation Metrics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-4">
          {metrics.compression_ratio !== undefined && (
            <div className="space-y-1">
              <div className="text-sm font-medium">Compression Ratio</div>
              <div className="text-lg font-bold text-green-600 dark:text-green-400">
                {(metrics.compression_ratio ?? 0).toFixed(2)}x
              </div>
            </div>
          )}
          
          {metrics.token_savings !== undefined && (
            <div className="space-y-1">
              <div className="text-sm font-medium">Token Savings</div>
              <div className="text-lg font-bold text-green-600 dark:text-green-400">
                {(metrics.token_savings ?? 0).toLocaleString()}
              </div>
            </div>
          )}
          
          {metrics.performance_impact !== undefined && (
            <div className="space-y-1">
              <div className="text-sm font-medium">Performance Impact</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                    style={{ width: `${Math.round((metrics.performance_impact ?? 0) * 100)}%` }}
                  />
                </div>
                <span className="text-sm font-medium">
                  {Math.round((metrics.performance_impact ?? 0) * 100)}%
                </span>
              </div>
            </div>
          )}
          
          {metrics.success_rate !== undefined && (
            <div className="space-y-1">
              <div className="text-sm font-medium">Success Rate</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                    style={{ width: `${Math.round((metrics.success_rate ?? 0) * 100)}%` }}
                  />
                </div>
                <span className="text-sm font-medium">
                  {Math.round((metrics.success_rate ?? 0) * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
        
        {metrics.session_id && (
          <div className="pt-2">
            <Separator className="mb-2" />
            <div className="text-xs text-muted-foreground">
              Session ID: {metrics.session_id}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ResearchProgressCard({
  progress,
}: {
  progress: NonNullable<Message["researchProgress"]>;
}) {
  return (
    <Card className="border-purple-200 bg-purple-50/50 dark:border-purple-800 dark:bg-purple-950/20">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-purple-700 dark:text-purple-300">
          <TrendingUp size={18} />
          Research Progress
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {progress.current_step && (
          <div>
            <div className="mb-1 text-sm font-medium">Current Step</div>
            <Badge variant="outline">{progress.current_step}</Badge>
          </div>
        )}
        
        {progress.completion_percentage !== undefined && (
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-medium">Progress</span>
              <span>{Math.round(progress.completion_percentage ?? 0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${progress.completion_percentage ?? 0}%` }}
              />
            </div>
          </div>
        )}
        
        <div className="flex items-center justify-between">
          {progress.status && (
            <Badge
              variant={progress.status === "completed" ? "default" : "secondary"}
            >
              {progress.status}
            </Badge>
          )}
          
          {progress.estimated_time_remaining !== undefined && (
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <Clock size={14} />
              {Math.round(progress.estimated_time_remaining)}s remaining
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function NodeUpdatesCard({
  updates,
}: {
  updates: NonNullable<Message["nodeUpdates"]>;
}) {
  return (
    <Card className="border-orange-200 bg-orange-50/50 dark:border-orange-800 dark:bg-orange-950/20">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-orange-700 dark:text-orange-300">
          <FileText size={18} />
          Node Updates ({updates.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {updates.slice(-3).map((update, index) => (
            <div
              key={index}
              className="rounded-md border bg-background/50 p-3 text-sm"
            >
              <div className="mb-1 flex items-center justify-between">
                <Badge variant="outline">{update.node}</Badge>
                <span className="text-xs text-muted-foreground">
                  {new Date(update.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="text-muted-foreground">
                Agent: {update.agent} • {Object.keys(update.data).length} data fields
              </div>
            </div>
          ))}
          {updates.length > 3 && (
            <div className="text-center text-xs text-muted-foreground">
              ... and {updates.length - 3} more updates
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}