#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeerFlow 并行化优化性能基准测试

此脚本用于测试和比较不同优化级别下的性能表现，
包括执行时间、资源利用率和结果质量的对比。
"""

import asyncio
import time
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入必要的模块
from src.workflow import (
    run_agent_workflow_async,
    run_optimized_research_workflow,
    run_parallel_report_generation,
)
from src.utils.workflow_optimizer import (
    WorkflowOptimizationLevel,
)
from src.utils.performance_optimizer import (
    get_global_advanced_executor,
    shutdown_global_executor,
)


class BenchmarkResult:
    """存储基准测试结果的数据类"""

    def __init__(
        self,
        optimization_level: str,
        execution_time: float,
        cpu_usage: float,
        memory_usage: float,
        task_count: int,
        parallel_tasks: int,
        metrics: Dict[str, Any],
    ):
        self.optimization_level = optimization_level
        self.execution_time = execution_time
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.task_count = task_count
        self.parallel_tasks = parallel_tasks
        self.metrics = metrics
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式"""
        return {
            "optimization_level": self.optimization_level,
            "execution_time": self.execution_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "task_count": self.task_count,
            "parallel_tasks": self.parallel_tasks,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """从字典创建结果对象"""
        return cls(
            optimization_level=data["optimization_level"],
            execution_time=data["execution_time"],
            cpu_usage=data["cpu_usage"],
            memory_usage=data["memory_usage"],
            task_count=data["task_count"],
            parallel_tasks=data["parallel_tasks"],
            metrics=data["metrics"],
        )


class ParallelOptimizationBenchmark:
    """并行优化性能基准测试类"""

    def __init__(
        self, save_results: bool = True, results_file: str = "benchmark_results.json"
    ):
        self.save_results = save_results
        self.results_file = results_file
        self.results = []

        # 尝试加载之前的结果
        if save_results and os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.results = [BenchmarkResult.from_dict(item) for item in data]
                print(f"已加载 {len(self.results)} 条历史基准测试结果")
            except Exception as e:
                print(f"加载历史结果失败: {e}")

    async def run_research_benchmark(
        self, query: str, optimization_levels: List[WorkflowOptimizationLevel] = None
    ) -> List[BenchmarkResult]:
        """运行研究工作流基准测试"""
        if optimization_levels is None:
            optimization_levels = [
                WorkflowOptimizationLevel.BASIC,
                WorkflowOptimizationLevel.STANDARD,
                WorkflowOptimizationLevel.ADVANCED,
                WorkflowOptimizationLevel.MAXIMUM,
            ]

        benchmark_results = []

        for level in optimization_levels:
            print(f"\n运行优化级别: {level.value} 的基准测试...")
            start_time = time.time()

            # 运行优化的研究工作流
            result = await run_optimized_research_workflow(
                user_input=query,
                workflow_type="research",
                optimization_level=level,
                enable_intelligent_task_decomposition=(
                    level.value in ["advanced", "maximum"]
                ),
                enable_dynamic_resource_allocation=(level.value == "maximum"),
                max_workers=8 if level.value in ["basic", "standard"] else 12,
            )

            execution_time = time.time() - start_time

            # 获取性能指标
            metrics = result.get("performance_metrics", {})

            # 创建基准测试结果
            benchmark_result = BenchmarkResult(
                optimization_level=level.value,
                execution_time=execution_time,
                cpu_usage=metrics.get("cpu_usage", 0.0),
                memory_usage=metrics.get("memory_usage", 0.0),
                task_count=metrics.get("total_tasks", 0),
                parallel_tasks=metrics.get("parallel_tasks", 0),
                metrics=metrics,
            )

            benchmark_results.append(benchmark_result)
            self.results.append(benchmark_result)

            print(f"完成 {level.value} 级别测试，执行时间: {execution_time:.2f}秒")
            print(
                f"CPU使用率: {benchmark_result.cpu_usage:.1f}%, 内存使用: {benchmark_result.memory_usage:.1f}MB"
            )
            print(
                f"任务总数: {benchmark_result.task_count}, 并行任务: {benchmark_result.parallel_tasks}"
            )

        # 保存结果
        if self.save_results:
            self._save_results()

        return benchmark_results

    async def run_report_generation_benchmark(
        self,
        content_sections: List[str],
        report_type: str = "comprehensive",
        optimization_levels: List[WorkflowOptimizationLevel] = None,
    ) -> List[BenchmarkResult]:
        """运行报告生成基准测试"""
        if optimization_levels is None:
            optimization_levels = [
                WorkflowOptimizationLevel.BASIC,
                WorkflowOptimizationLevel.STANDARD,
                WorkflowOptimizationLevel.ADVANCED,
                WorkflowOptimizationLevel.MAXIMUM,
            ]

        benchmark_results = []

        for level in optimization_levels:
            print(f"\n运行报告生成优化级别: {level.value} 的基准测试...")
            start_time = time.time()

            # 运行并行报告生成
            result = await run_parallel_report_generation(
                content_sections=content_sections,
                report_type=report_type,
                user_context="基准测试",
                optimization_level=level,
            )

            execution_time = time.time() - start_time

            # 获取性能指标
            metrics = result.get("performance_metrics", {})

            # 创建基准测试结果
            benchmark_result = BenchmarkResult(
                optimization_level=level.value,
                execution_time=execution_time,
                cpu_usage=metrics.get("cpu_usage", 0.0),
                memory_usage=metrics.get("memory_usage", 0.0),
                task_count=len(content_sections),
                parallel_tasks=metrics.get("parallel_sections", 0),
                metrics=metrics,
            )

            benchmark_results.append(benchmark_result)
            self.results.append(benchmark_result)

            print(
                f"完成报告生成 {level.value} 级别测试，执行时间: {execution_time:.2f}秒"
            )
            print(
                f"CPU使用率: {benchmark_result.cpu_usage:.1f}%, 内存使用: {benchmark_result.memory_usage:.1f}MB"
            )

        # 保存结果
        if self.save_results:
            self._save_results()

        return benchmark_results

    async def run_comparison_benchmark(self, query: str) -> Dict[str, Any]:
        """运行传统工作流与优化工作流的比较基准测试"""
        print("\n开始比较传统工作流与优化工作流性能差异...")

        # 传统工作流（无优化）
        print("\n运行传统工作流（无优化）...")
        start_time = time.time()
        traditional_result = await run_agent_workflow_async(
            user_input=query,
            enable_advanced_optimization=False,
            enable_hierarchical_memory=False,
            enable_adaptive_rate_limiting=False,
        )
        traditional_time = time.time() - start_time

        # 最大优化工作流
        print("\n运行最大优化工作流...")
        start_time = time.time()
        optimized_result = await run_agent_workflow_async(
            user_input=query,
            enable_advanced_optimization=True,
            enable_hierarchical_memory=True,
            enable_adaptive_rate_limiting=True,
            optimization_level=WorkflowOptimizationLevel.MAXIMUM,
            enable_intelligent_task_decomposition=True,
            enable_dynamic_resource_allocation=True,
        )
        optimized_time = time.time() - start_time

        # 计算性能提升
        time_improvement = (
            (traditional_time - optimized_time) / traditional_time
        ) * 100

        # 获取性能指标
        traditional_metrics = traditional_result.get("performance_metrics", {})
        optimized_metrics = optimized_result.get("performance_metrics", {})

        comparison = {
            "traditional_execution_time": traditional_time,
            "optimized_execution_time": optimized_time,
            "time_improvement_percentage": time_improvement,
            "traditional_metrics": traditional_metrics,
            "optimized_metrics": optimized_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        print("\n性能比较结果:")
        print(f"传统工作流执行时间: {traditional_time:.2f}秒")
        print(f"优化工作流执行时间: {optimized_time:.2f}秒")
        print(f"性能提升: {time_improvement:.1f}%")

        # 保存比较结果
        if self.save_results:
            comparison_file = "workflow_comparison.json"
            try:
                with open(comparison_file, "w", encoding="utf-8") as f:
                    json.dump(comparison, f, indent=2, ensure_ascii=False)
                print(f"比较结果已保存到 {comparison_file}")
            except Exception as e:
                print(f"保存比较结果失败: {e}")

        return comparison

    def generate_report(self) -> str:
        """生成基准测试报告"""
        if not self.results:
            return "没有可用的基准测试结果"

        report = ["# DeerFlow 并行化优化基准测试报告\n"]
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 按优化级别分组
        results_by_level = {}
        for result in self.results:
            level = result.optimization_level
            if level not in results_by_level:
                results_by_level[level] = []
            results_by_level[level].append(result)

        # 计算每个级别的平均值
        report.append("## 优化级别性能对比\n")
        report.append(
            "| 优化级别 | 平均执行时间(秒) | CPU使用率(%) | 内存使用(MB) | 并行任务数 |"
        )
        report.append("| --- | --- | --- | --- | --- |")

        for level in ["basic", "standard", "advanced", "maximum"]:
            if level in results_by_level:
                level_results = results_by_level[level]
                avg_time = sum(r.execution_time for r in level_results) / len(
                    level_results
                )
                avg_cpu = sum(r.cpu_usage for r in level_results) / len(level_results)
                avg_memory = sum(r.memory_usage for r in level_results) / len(
                    level_results
                )
                avg_parallel = sum(r.parallel_tasks for r in level_results) / len(
                    level_results
                )

                report.append(
                    f"| {level.capitalize()} | {avg_time:.2f} | {avg_cpu:.1f} | {avg_memory:.1f} | {avg_parallel:.1f} |"
                )

        # 性能提升分析
        if "basic" in results_by_level and "maximum" in results_by_level:
            basic_avg_time = sum(
                r.execution_time for r in results_by_level["basic"]
            ) / len(results_by_level["basic"])
            max_avg_time = sum(
                r.execution_time for r in results_by_level["maximum"]
            ) / len(results_by_level["maximum"])
            time_improvement = ((basic_avg_time - max_avg_time) / basic_avg_time) * 100

            report.append("\n## 性能提升分析\n")
            report.append(
                f"* 基础优化到最大优化的执行时间减少: **{time_improvement:.1f}%**"
            )
            report.append(f"* 基础优化平均执行时间: {basic_avg_time:.2f}秒")
            report.append(f"* 最大优化平均执行时间: {max_avg_time:.2f}秒")

        # 最近测试结果详情
        recent_results = sorted(self.results, key=lambda x: x.timestamp, reverse=True)[
            :5
        ]
        if recent_results:
            report.append("\n## 最近测试结果详情\n")
            for i, result in enumerate(recent_results):
                report.append(
                    f"### 测试 {i+1} ({result.optimization_level.capitalize()})\n"
                )
                report.append(f"* 执行时间: {result.execution_time:.2f}秒")
                report.append(f"* CPU使用率: {result.cpu_usage:.1f}%")
                report.append(f"* 内存使用: {result.memory_usage:.1f}MB")
                report.append(f"* 任务总数: {result.task_count}")
                report.append(f"* 并行任务数: {result.parallel_tasks}")
                report.append(f"* 时间戳: {result.timestamp}\n")

        return "\n".join(report)

    def _save_results(self) -> None:
        """保存基准测试结果到文件"""
        try:
            results_data = [result.to_dict() for result in self.results]
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"基准测试结果已保存到 {self.results_file}")
        except Exception as e:
            print(f"保存结果失败: {e}")

    def save_report(self, filename: str = "benchmark_report.md") -> None:
        """将报告保存到文件"""
        report = self.generate_report()
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"基准测试报告已保存到 {filename}")
        except Exception as e:
            print(f"保存报告失败: {e}")


async def main():
    """主函数"""
    benchmark = ParallelOptimizationBenchmark()

    print("=== DeerFlow 并行化优化性能基准测试 ===")
    print("1. 研究工作流基准测试")
    print("2. 报告生成基准测试")
    print("3. 工作流比较基准测试")
    print("4. 生成基准测试报告")
    print("5. 运行所有测试")
    print("0. 退出")

    choice = input("\n请选择测试类型 [0-5]: ")

    if choice == "1":
        query = (
            input("请输入研究查询: ") or "分析人工智能在医疗领域的应用和未来发展趋势"
        )
        levels = [WorkflowOptimizationLevel.BASIC, WorkflowOptimizationLevel.MAXIMUM]
        await benchmark.run_research_benchmark(query, levels)

    elif choice == "2":
        sections = [
            "执行摘要",
            "市场分析",
            "技术评估",
            "风险分析",
            "未来展望",
            "建议与结论",
        ]
        await benchmark.run_report_generation_benchmark(sections)

    elif choice == "3":
        query = input("请输入比较查询: ") or "分析区块链技术在供应链管理中的应用"
        await benchmark.run_comparison_benchmark(query)

    elif choice == "4":
        report = benchmark.generate_report()
        print("\n" + report)
        benchmark.save_report()

    elif choice == "5":
        # 运行所有测试
        query = "分析人工智能在医疗领域的应用和未来发展趋势"
        levels = [WorkflowOptimizationLevel.BASIC, WorkflowOptimizationLevel.MAXIMUM]
        await benchmark.run_research_benchmark(query, levels)

        sections = ["执行摘要", "市场分析", "技术评估", "风险分析"]
        await benchmark.run_report_generation_benchmark(sections)

        await benchmark.run_comparison_benchmark("分析区块链技术在供应链管理中的应用")

        benchmark.save_report()

    else:
        print("退出程序")

    # 确保关闭全局执行器
    executor = get_global_advanced_executor()
    if executor:
        await shutdown_global_executor()


if __name__ == "__main__":
    asyncio.run(main())
