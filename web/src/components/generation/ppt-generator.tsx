"use client";

import { useState } from "react";
import {
  FileText,
  Loader2,
  Presentation,
  Sparkles,
  Download,
  Copy,
  Check,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Textarea } from "~/components/ui/textarea";

interface PPTGeneratorProps {
  className?: string;
  onGenerated?: (blob: Blob) => void;
}

export function PPTGenerator({
  className = "",
  onGenerated,
}: PPTGeneratorProps) {
  const [content, setContent] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pptBlob, setPptBlob] = useState<Blob | null>(null);
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    if (!content.trim()) {
      setError("Please enter content for the presentation");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setPptBlob(null);

    try {
      const response = await fetch("/api/ppt/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate PPT: ${response.statusText}`);
      }

      const blob = await response.blob();
      setPptBlob(blob);
      onGenerated?.(blob);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!pptBlob) return;

    const url = URL.createObjectURL(pptBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `presentation-${Date.now()}.pptx`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyContent = async () => {
    if (!content) return;
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const wordCount = content.trim().split(/\s+/).filter(Boolean).length;

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Presentation className="h-5 w-5" />
            PPT Generator
          </CardTitle>
          <Badge variant="outline">{wordCount} words</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Content</label>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopyContent}
              disabled={!content}
            >
              {copied ? (
                <Check className="h-4 w-4" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          </div>
          <Textarea
            placeholder="Enter the content for your presentation. This will be used to generate slides automatically..."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="min-h-[200px] font-mono text-sm"
          />
        </div>

        {error && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="flex items-center gap-2">
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !content.trim()}
            className="flex-1"
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate PPT
              </>
            )}
          </Button>

          {pptBlob && (
            <Button variant="outline" onClick={handleDownload}>
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          )}
        </div>

        {pptBlob && (
          <div className="rounded-md bg-green-500/10 p-3 text-sm text-green-600">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Presentation generated successfully! Click Download to save.
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Size: {(pptBlob.size / 1024).toFixed(1)} KB
            </div>
          </div>
        )}

        <div className="text-xs text-muted-foreground">
          <p className="font-medium mb-1">Tips:</p>
          <ul className="list-disc list-inside space-y-1">
            <li>Use clear headings and bullet points</li>
            <li>Include key statistics and data</li>
            <li>Structure content with sections for better slide organization</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
