"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Bot,
  Check,
  ChevronDown,
  Cog,
  Loader2,
  RefreshCw,
  Sparkles,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { ScrollArea } from "~/components/ui/scroll-area";
import { cn } from "~/lib/utils";

interface ModelInfo {
  provider: string;
  models: string[];
}

interface ConfigResponse {
  rag: {
    provider: string;
  };
  models: Record<string, string[]>;
}

interface ModelSelectorProps {
  value?: string;
  onChange?: (model: string, provider: string) => void;
  className?: string;
  showDetails?: boolean;
}

export function ModelSelector({
  value,
  onChange,
  className = "",
  showDetails = false,
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>(value || "");
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/config");
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

      const data: ConfigResponse = await response.json();
      const modelList: ModelInfo[] = Object.entries(data.models).map(
        ([provider, modelList]) => ({
          provider,
          models: modelList,
        })
      );

      setModels(modelList);

      if (!selectedModel && modelList.length > 0 && modelList[0] && modelList[0].models.length > 0) {
        const firstModel = modelList[0];
        const firstModelName = firstModel.models[0];
        if (firstModelName) {
          setSelectedModel(firstModelName);
          setSelectedProvider(firstModel.provider);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }, [selectedModel]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  useEffect(() => {
    if (value) {
      setSelectedModel(value);
      for (const info of models) {
        if (info.models.includes(value)) {
          setSelectedProvider(info.provider);
          break;
        }
      }
    }
  }, [value, models]);

  const handleSelect = (model: string, provider: string) => {
    setSelectedModel(model);
    setSelectedProvider(provider);
    setOpen(false);
    onChange?.(model, provider);
  };

  const getProviderIcon = (provider: string) => {
    const lower = provider.toLowerCase();
    if (lower.includes("openai")) return "🟢";
    if (lower.includes("anthropic")) return "🟣";
    if (lower.includes("google")) return "🔵";
    if (lower.includes("deepseek")) return "🟡";
    if (lower.includes("ollama")) return "🦙";
    return "🤖";
  };

  const getModelDisplayName = (model: string) => {
    const parts = model.split("/");
    return parts.length > 1 ? parts[parts.length - 1] : model;
  };

  const totalModels = models.reduce((acc, m) => acc + m.models.length, 0);

  if (!showDetails) {
    return (
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={cn("justify-between min-w-[200px]", className)}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : selectedModel ? (
              <span className="truncate">
                {getProviderIcon(selectedProvider)} {getModelDisplayName(selectedModel)}
              </span>
            ) : (
              "Select model..."
            )}
            <ChevronDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[300px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search models..." />
            <CommandList>
              <CommandEmpty>No models found.</CommandEmpty>
              {models.map((info) => (
                <CommandGroup key={info.provider} heading={info.provider}>
                  <ScrollArea className="max-h-[200px]">
                    {info.models.map((model) => (
                      <CommandItem
                        key={model}
                        value={`${info.provider}-${model}`}
                        onSelect={() => handleSelect(model, info.provider)}
                      >
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            selectedModel === model
                              ? "opacity-100"
                              : "opacity-0"
                          )}
                        />
                        <span className="mr-2">{getProviderIcon(info.provider)}</span>
                        {getModelDisplayName(model)}
                      </CommandItem>
                    ))}
                  </ScrollArea>
                </CommandGroup>
              ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    );
  }

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Bot className="h-5 w-5" />
            Model Selection
          </CardTitle>
          <Button
            variant="ghost"
            size="icon"
            onClick={fetchModels}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {totalModels} models available from {models.length} providers
          </span>
        </div>

        <ScrollArea className="h-[300px] pr-4">
          <div className="space-y-4">
            {models.map((info) => (
              <div key={info.provider} className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{getProviderIcon(info.provider)}</span>
                  <span className="font-medium">{info.provider}</span>
                  <Badge variant="outline" className="text-xs">
                    {info.models.length} models
                  </Badge>
                </div>
                <div className="grid gap-1 pl-7">
                  {info.models.map((model) => (
                    <div
                      key={model}
                      className={cn(
                        "flex items-center justify-between p-2 rounded-md cursor-pointer transition-colors",
                        selectedModel === model
                          ? "bg-primary/10 border border-primary/20"
                          : "hover:bg-muted"
                      )}
                      onClick={() => handleSelect(model, info.provider)}
                    >
                      <div className="flex items-center gap-2">
                        {selectedModel === model && (
                          <Check className="h-4 w-4 text-primary" />
                        )}
                        <span className="text-sm font-mono">
                          {getModelDisplayName(model)}
                        </span>
                      </div>
                      <Sparkles className="h-3 w-3 text-muted-foreground" />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {selectedModel && (
          <div className="pt-2 border-t">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Selected:</span>
              <div className="flex items-center gap-2">
                <span>{getProviderIcon(selectedProvider)}</span>
                <code className="text-sm bg-muted px-2 py-1 rounded">
                  {getModelDisplayName(selectedModel)}
                </code>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
