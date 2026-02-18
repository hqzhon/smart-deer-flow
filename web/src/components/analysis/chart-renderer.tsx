"use client";

import { useState, useCallback } from "react";
import {
  BarChart3,
  LineChart,
  PieChart,
  ScatterChart,
  TrendingUp,
  Download,
  RefreshCw,
  Loader2,
} from "lucide-react";

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";

import {
  createChart,
  calculateStatistics,
  type ChartResponse,
  type StatisticsResponse,
} from "~/core/api/analysis";

interface ChartData {
  type: string;
  title: string;
  imageBase64?: string;
  html?: string;
  data?: Record<string, unknown>;
}

interface AnalysisResult {
  statistics?: Record<string, StatisticsResponse>;
  correlation?: Record<string, Record<string, number>>;
  outliers?: Record<string, unknown>;
  distributions?: Record<string, Record<string, number>>;
}

interface DisplayChart {
  title?: string;
  imageBase64?: string;
  html?: string;
  data?: Record<string, unknown>;
}

interface ChartRendererProps {
  chartData?: ChartData;
  analysisResult?: AnalysisResult;
  rawData?: Array<number>;
  labels?: Array<string>;
  className?: string;
  onChartCreated?: (chart: ChartResponse) => void;
}

export function ChartRenderer({
  chartData,
  analysisResult,
  rawData,
  labels,
  className = "",
  onChartCreated,
}: ChartRendererProps) {
  const [chartType, setChartType] = useState<string>("bar");
  const [isLoading, setIsLoading] = useState(false);
  const [generatedChart, setGeneratedChart] = useState<ChartResponse | null>(null);
  const [statistics, setStatistics] = useState<StatisticsResponse | null>(null);

  const chartTypes = [
    { value: "line", label: "Line Chart", icon: LineChart },
    { value: "bar", label: "Bar Chart", icon: BarChart3 },
    { value: "pie", label: "Pie Chart", icon: PieChart },
    { value: "scatter", label: "Scatter Plot", icon: ScatterChart },
    { value: "histogram", label: "Histogram", icon: BarChart3 },
    { value: "area", label: "Area Chart", icon: TrendingUp },
  ];

  const handleGenerateChart = useCallback(async () => {
    if (!rawData || rawData.length === 0) return;

    setIsLoading(true);
    try {
      const response = await createChart({
        chart_type: chartType as "line" | "bar" | "pie" | "scatter" | "histogram" | "area",
        title: "Generated Chart",
        data: rawData,
        labels: labels,
      });
      setGeneratedChart(response);
      onChartCreated?.(response);
    } catch (err) {
      console.error("Failed to generate chart:", err);
    } finally {
      setIsLoading(false);
    }
  }, [rawData, labels, chartType, onChartCreated]);

  const handleCalculateStats = useCallback(async () => {
    if (!rawData || rawData.length === 0) return;

    setIsLoading(true);
    try {
      const stats = await calculateStatistics(rawData, true);
      setStatistics(stats);
    } catch (err) {
      console.error("Failed to calculate statistics:", err);
    } finally {
      setIsLoading(false);
    }
  }, [rawData]);

  const handleDownload = () => {
    const imageBase64 = chartData?.imageBase64 || (generatedChart?.chart_data?.base64_image as string | undefined);
    if (!imageBase64) return;

    const link = document.createElement("a");
    link.href = `data:image/png;base64,${imageBase64}`;
    link.download = `chart-${Date.now()}.png`;
    link.click();
  };

  const handleExportData = () => {
    const data = chartData?.data || generatedChart?.chart_data || rawData;
    if (!data) return;

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `chart-data-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const displayChart: DisplayChart | null = generatedChart
    ? {
        title: generatedChart.title,
        imageBase64: generatedChart.chart_data?.base64_image as string | undefined,
        html: generatedChart.chart_data?.html as string | undefined,
        data: generatedChart.chart_data,
      }
    : chartData || null;
  const displayStats = statistics || (analysisResult?.statistics ? Object.values(analysisResult.statistics)[0] : null);

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <BarChart3 className="h-5 w-5" />
            Data Visualization
          </CardTitle>
          <div className="flex items-center gap-2">
            <Select value={chartType} onValueChange={setChartType}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Chart type" />
              </SelectTrigger>
              <SelectContent>
                {chartTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    <div className="flex items-center gap-2">
                      <type.icon className="h-4 w-4" />
                      {type.label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="chart" className="w-full">
          <TabsList className="mb-2">
            <TabsTrigger value="chart">Chart</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="data">Data</TabsTrigger>
          </TabsList>

          <TabsContent value="chart" className="space-y-4">
            <div className="relative aspect-video overflow-hidden rounded-md border bg-muted">
              {displayChart?.imageBase64 ? (
                <img
                  src={`data:image/png;base64,${displayChart.imageBase64}`}
                  alt={displayChart.title || "Chart"}
                  className="h-full w-full object-contain"
                />
              ) : displayChart?.html ? (
                <iframe
                  srcDoc={displayChart.html}
                  className="h-full w-full border-0"
                  title="Chart"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <BarChart3 className="mx-auto h-12 w-12 opacity-50" />
                    <p className="mt-2 text-sm">
                      {rawData ? "Click Generate to create chart" : "No chart data available"}
                    </p>
                  </div>
                </div>
              )}

              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              )}
            </div>

            {(displayChart?.title || generatedChart?.title) && (
              <div className="text-center text-sm font-medium">
                {displayChart?.title || generatedChart?.title}
              </div>
            )}

            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleGenerateChart}
                disabled={!rawData || isLoading}
              >
                {isLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="mr-2 h-4 w-4" />
                )}
                Generate
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                disabled={!displayChart?.imageBase64}
              >
                <Download className="mr-2 h-4 w-4" />
                Download
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="space-y-4">
            <div className="flex justify-end mb-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCalculateStats}
                disabled={!rawData || isLoading}
              >
                {isLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <TrendingUp className="mr-2 h-4 w-4" />
                )}
                Calculate Stats
              </Button>
            </div>

            {displayStats ? (
              <div className="space-y-4">
                <div className="rounded-md border p-3">
                  <div className="mb-2 flex items-center gap-2 font-medium">
                    <TrendingUp className="h-4 w-4" />
                    Descriptive Statistics
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                      <span className="text-muted-foreground">Count:</span>{" "}
                      <span className="font-medium">{displayStats.count}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Mean:</span>{" "}
                      <span className="font-medium">{displayStats.mean.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Median:</span>{" "}
                      <span className="font-medium">{displayStats.median.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Std:</span>{" "}
                      <span className="font-medium">{displayStats.std.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Min:</span>{" "}
                      <span className="font-medium">{displayStats.min.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Max:</span>{" "}
                      <span className="font-medium">{displayStats.max.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Range:</span>{" "}
                      <span className="font-medium">{displayStats.range.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Sum:</span>{" "}
                      <span className="font-medium">{displayStats.sum.toFixed(4)}</span>
                    </div>
                  </div>
                </div>

                {displayStats.quartiles && (
                  <div className="rounded-md border p-3">
                    <div className="mb-2 font-medium">Quartiles</div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div>
                        <span className="text-muted-foreground">Q1:</span>{" "}
                        <span className="font-medium">{displayStats.quartiles.q1.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Q2 (Median):</span>{" "}
                        <span className="font-medium">{displayStats.quartiles.q2.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Q3:</span>{" "}
                        <span className="font-medium">{displayStats.quartiles.q3.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">IQR:</span>{" "}
                        <span className="font-medium">{displayStats.quartiles.iqr.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                )}

                {displayStats.percentiles && (
                  <div className="rounded-md border p-3">
                    <div className="mb-2 font-medium">Percentiles</div>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(displayStats.percentiles).map(([key, value]) => (
                        <Badge key={key} variant="outline">
                          {key.toUpperCase()}: {value.toFixed(4)}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : analysisResult ? (
              <div className="space-y-4">
                {analysisResult.statistics && (
                  <div className="rounded-md border p-3">
                    <div className="mb-2 flex items-center gap-2 font-medium">
                      <TrendingUp className="h-4 w-4" />
                      Descriptive Statistics
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="p-2 text-left">Column</th>
                            <th className="p-2 text-right">Mean</th>
                            <th className="p-2 text-right">Std</th>
                            <th className="p-2 text-right">Min</th>
                            <th className="p-2 text-right">Max</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(analysisResult.statistics).map(
                            ([col, stats]) => (
                              <tr key={col} className="border-b">
                                <td className="p-2 font-medium">{col}</td>
                                <td className="p-2 text-right">
                                  {stats.mean?.toFixed(2)}
                                </td>
                                <td className="p-2 text-right">
                                  {stats.std?.toFixed(2)}
                                </td>
                                <td className="p-2 text-right">
                                  {stats.min?.toFixed(2)}
                                </td>
                                <td className="p-2 text-right">
                                  {stats.max?.toFixed(2)}
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {analysisResult.correlation && (
                  <div className="rounded-md border p-3">
                    <div className="mb-2 font-medium">Correlation Matrix</div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr>
                            <th className="p-2"></th>
                            {Object.keys(analysisResult.correlation).map(
                              (col) => (
                                <th key={col} className="p-2 text-xs">
                                  {col}
                                </th>
                              )
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(analysisResult.correlation).map(
                            ([row, cols]) => (
                              <tr key={row}>
                                <td className="p-2 text-xs font-medium">
                                  {row}
                                </td>
                                {Object.values(cols).map((val, i) => (
                                  <td
                                    key={i}
                                    className="p-2 text-center text-xs"
                                  >
                                    <Badge
                                      variant={
                                        Math.abs(val as number) > 0.7
                                          ? "default"
                                          : "outline"
                                      }
                                      className="text-xs"
                                    >
                                      {(val as number).toFixed(2)}
                                    </Badge>
                                  </td>
                                ))}
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {analysisResult.outliers && (
                  <div className="rounded-md border p-3">
                    <div className="mb-2 font-medium">Outliers Detected</div>
                    <div className="space-y-2">
                      {Object.entries(analysisResult.outliers).map(
                        ([col, info]) => (
                          <div
                            key={col}
                            className="flex items-center justify-between text-sm"
                          >
                            <span>{col}</span>
                            <Badge variant="outline">
                              {(info as { count: number }).count} outliers
                            </Badge>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                {rawData ? "Click Calculate Stats to analyze data" : "No analysis results available"}
              </div>
            )}
          </TabsContent>

          <TabsContent value="data">
            <div className="flex justify-end mb-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportData}
                disabled={!rawData && !displayChart?.data}
              >
                <Download className="mr-2 h-4 w-4" />
                Export JSON
              </Button>
            </div>

            {(displayChart?.data || rawData) ? (
              <div className="rounded-md border p-3">
                <pre className="overflow-x-auto text-xs max-h-[400px]">
                  {JSON.stringify(displayChart?.data || rawData, null, 2)}
                </pre>
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
