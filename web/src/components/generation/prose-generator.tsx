"use client";

import { useState, useRef, useEffect } from "react";
import {
  BookOpen,
  Copy,
  Check,
  Loader2,
  PenTool,
  Sparkles,
  StopCircle,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Textarea } from "~/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Input } from "~/components/ui/input";

const PROSE_OPTIONS = [
  { value: "expand", label: "Expand" },
  { value: "rewrite", label: "Rewrite" },
  { value: "summarize", label: "Summarize" },
  { value: "polish", label: "Polish" },
  { value: "translate", label: "Translate" },
];

interface ProseGeneratorProps {
  className?: string;
  onGenerated?: (text: string) => void;
}

export function ProseGenerator({
  className = "",
  onGenerated,
}: ProseGeneratorProps) {
  const [prompt, setPrompt] = useState("");
  const [option, setOption] = useState("expand");
  const [command, setCommand] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (textareaRef.current && generatedText) {
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
    }
  }, [generatedText]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setGeneratedText("");

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch("/api/prose/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          option,
          command: command || undefined,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Failed to generate prose: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      const decoder = new TextDecoder();
      let accumulatedText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const content = line.slice(6);
            accumulatedText += content;
            setGeneratedText(accumulatedText);
          }
        }
      }

      onGenerated?.(accumulatedText);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
    setIsGenerating(false);
  };

  const handleCopy = async () => {
    if (!generatedText) return;
    await navigator.clipboard.writeText(generatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const wordCount = prompt.trim().split(/\s+/).filter(Boolean).length;

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <PenTool className="h-5 w-5" />
            Prose Generator
          </CardTitle>
          <Badge variant="outline">{wordCount} words</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Prompt</label>
          <Textarea
            placeholder="Enter your text or prompt for prose generation..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="min-h-[120px] font-mono text-sm"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Option</label>
            <Select value={option} onValueChange={setOption}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PROSE_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Custom Command (Optional)</label>
            <Input
              placeholder="e.g., 'in the style of Shakespeare'"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
            />
          </div>
        </div>

        {error && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="flex items-center gap-2">
          {isGenerating ? (
            <Button variant="destructive" onClick={handleStop} className="flex-1">
              <StopCircle className="mr-2 h-4 w-4" />
              Stop
            </Button>
          ) : (
            <Button
              onClick={handleGenerate}
              disabled={!prompt.trim()}
              className="flex-1"
            >
              <Sparkles className="mr-2 h-4 w-4" />
              Generate
            </Button>
          )}

          {generatedText && (
            <Button variant="outline" onClick={handleCopy}>
              {copied ? (
                <Check className="h-4 w-4" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>

        {generatedText && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium flex items-center gap-2">
                <BookOpen className="h-4 w-4" />
                Generated Text
              </label>
              <Badge variant="secondary">
                {generatedText.split(/\s+/).filter(Boolean).length} words
              </Badge>
            </div>
            <div
              ref={textareaRef}
              className="rounded-md border bg-muted p-3 max-h-[300px] overflow-y-auto"
            >
              <pre className="whitespace-pre-wrap font-mono text-sm">
                {generatedText}
              </pre>
            </div>
          </div>
        )}

        <div className="text-xs text-muted-foreground">
          <p className="font-medium mb-1">Options:</p>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>Expand:</strong> Elaborate and add more detail</li>
            <li><strong>Rewrite:</strong> Rephrase the content</li>
            <li><strong>Summarize:</strong> Create a concise summary</li>
            <li><strong>Polish:</strong> Improve writing quality</li>
            <li><strong>Translate:</strong> Convert to another language</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
