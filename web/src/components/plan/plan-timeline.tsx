"use client";

import { useState } from "react";
import {
  CheckCircle2,
  Circle,
  Clock,
  Loader2,
  XCircle,
  ChevronDown,
  ChevronUp,
  User,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { ScrollArea } from "~/components/ui/scroll-area";

export type StepStatus = "not_started" | "in_progress" | "completed" | "blocked" | "skipped";

export interface PlanStep {
  id: string;
  title: string;
  description?: string;
  status: StepStatus;
  agent?: string;
  dependencies?: string[];
  estimated_time?: string;
  started_at?: string;
  completed_at?: string;
  result?: string;
  error?: string;
}

export interface Plan {
  id: string;
  title: string;
  description?: string;
  steps: PlanStep[];
  created_at: string;
  updated_at: string;
  current_step_index: number;
  progress: number;
}

interface StepCardProps {
  step: PlanStep;
  index: number;
  isCurrent: boolean;
  isExpanded: boolean;
  onToggle: () => void;
  onClick?: () => void;
}

function StepCard({
  step,
  index,
  isCurrent,
  isExpanded,
  onToggle,
  onClick,
}: StepCardProps) {
  const statusConfig = {
    not_started: {
      icon: Circle,
      color: "text-muted-foreground",
      bgColor: "bg-muted",
      label: "Not Started",
    },
    in_progress: {
      icon: Loader2,
      color: "text-primary",
      bgColor: "bg-primary/10",
      label: "In Progress",
    },
    completed: {
      icon: CheckCircle2,
      color: "text-green-500",
      bgColor: "bg-green-500/10",
      label: "Completed",
    },
    blocked: {
      icon: XCircle,
      color: "text-destructive",
      bgColor: "bg-destructive/10",
      label: "Blocked",
    },
    skipped: {
      icon: Circle,
      color: "text-muted-foreground",
      bgColor: "bg-muted",
      label: "Skipped",
    },
  };

  const config = statusConfig[step.status];
  const StatusIcon = config.icon;

  return (
    <div
      className={`rounded-lg border transition-all ${
        isCurrent ? "border-primary ring-2 ring-primary/20" : ""
      } ${step.status === "in_progress" ? "animate-pulse-subtle" : ""}`}
    >
      <div
        className={`flex items-start gap-3 p-3 cursor-pointer hover:bg-muted/50`}
        onClick={onClick}
      >
        <div
          className={`flex h-8 w-8 items-center justify-center rounded-full ${config.bgColor}`}
        >
          {step.status === "in_progress" ? (
            <StatusIcon className={`h-4 w-4 ${config.color} animate-spin`} />
          ) : (
            <StatusIcon className={`h-4 w-4 ${config.color}`} />
          )}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Step {index + 1}</span>
            <Badge variant="outline" className="text-xs">
              {config.label}
            </Badge>
            {step.agent && (
              <Badge variant="secondary" className="text-xs">
                <User className="h-3 w-3 mr-1" />
                {step.agent}
              </Badge>
            )}
          </div>
          <h4 className="font-medium mt-1 truncate">{step.title}</h4>
          {step.description && (
            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
              {step.description}
            </p>
          )}
        </div>

        <div className="flex items-center gap-2">
          {step.estimated_time && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              {step.estimated_time}
            </div>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            className="p-1 hover:bg-muted rounded"
          >
            {isExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="px-3 pb-3 pt-0 border-t">
          {step.dependencies && step.dependencies.length > 0 && (
            <div className="mt-2 text-xs">
              <span className="text-muted-foreground">Depends on: </span>
              {step.dependencies.map((dep, i) => (
                <Badge key={dep} variant="outline" className="mr-1">
                  {dep}
                </Badge>
              ))}
            </div>
          )}

          {step.started_at && (
            <div className="mt-2 text-xs text-muted-foreground">
              Started: {new Date(step.started_at).toLocaleString()}
            </div>
          )}

          {step.completed_at && (
            <div className="text-xs text-muted-foreground">
              Completed: {new Date(step.completed_at).toLocaleString()}
            </div>
          )}

          {step.result && (
            <div className="mt-2 p-2 rounded bg-muted text-xs">
              <div className="font-medium mb-1">Result:</div>
              <pre className="whitespace-pre-wrap">{step.result}</pre>
            </div>
          )}

          {step.error && (
            <div className="mt-2 p-2 rounded bg-destructive/10 text-destructive text-xs">
              <div className="font-medium mb-1">Error:</div>
              <pre className="whitespace-pre-wrap">{step.error}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface PlanTimelineProps {
  plan: Plan;
  onStepClick?: (step: PlanStep, index: number) => void;
  className?: string;
}

export function PlanTimeline({
  plan,
  onStepClick,
  className = "",
}: PlanTimelineProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  const toggleStep = (stepId: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  };

  const completedSteps = plan.steps.filter(
    (s) => s.status === "completed"
  ).length;
  const totalSteps = plan.steps.length;
  const progressPercent = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{plan.title}</CardTitle>
          <Badge variant="outline">
            {completedSteps}/{totalSteps} completed
          </Badge>
        </div>
        {plan.description && (
          <p className="text-sm text-muted-foreground">{plan.description}</p>
        )}
        <div className="mt-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
            <span>Progress</span>
            <span>{progressPercent.toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px]">
          <div className="relative">
            {plan.steps.map((step, index) => (
              <div key={step.id} className="relative">
                {index < plan.steps.length - 1 && (
                  <div
                    className={`absolute left-4 top-10 w-0.5 h-full -mb-6 ${
                      step.status === "completed"
                        ? "bg-green-500"
                        : "bg-muted"
                    }`}
                  />
                )}
                <div className="relative pb-4">
                  <StepCard
                    step={step}
                    index={index}
                    isCurrent={index === plan.current_step_index}
                    isExpanded={expandedSteps.has(step.id)}
                    onToggle={() => toggleStep(step.id)}
                    onClick={() => onStepClick?.(step, index)}
                  />
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
