#!/usr/bin/env python3
"""并行化处理优化演示脚本

这个脚本展示了如何使用 DeerFlow 的并行化处理优化功能，
包括智能任务分解、动态资源分配和自适应负载均衡。
"""

import asyncio
import logging
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from src.workflow import (
        run_optimized_research_workflow,
        run_parallel_report_generation,
        run_agent_workflow_async,
    )
    from src.utils.performance.workflow_optimizer import (
        WorkflowOptimizationLevel,
    )
    from src.utils.performance.performance_optimizer import optimize_report_generation_workflow
except ImportError as e:
    logger.error(f"Failed to import DeerFlow modules: {e}")
    logger.info(
        "Please ensure you're running this script from the DeerFlow root directory"
    )
    exit(1)


async def demo_basic_optimization():
    """演示基础并行化优化"""
    logger.info("=== 基础并行化优化演示 ===")

    user_query = "分析人工智能在医疗领域的应用现状和发展趋势"

    start_time = time.time()

    try:
        # 使用基础优化级别
        result = await run_optimized_research_workflow(
            user_input=user_query,
            workflow_type="research",
            optimization_level=WorkflowOptimizationLevel.BASIC,
            enable_parallel_tasks=True,
            max_workers=4,
        )

        execution_time = time.time() - start_time

        logger.info(f"基础优化完成，耗时: {execution_time:.2f}秒")
        logger.info(f"优化应用状态: {result.get('optimization_applied', False)}")

        if "workflow_metrics" in result:
            metrics = result["workflow_metrics"]
            logger.info(f"性能指标: {metrics}")

        return result

    except Exception as e:
        logger.error(f"基础优化演示失败: {e}")
        return None


async def demo_advanced_optimization():
    """演示高级并行化优化"""
    logger.info("=== 高级并行化优化演示 ===")

    user_query = "深度分析区块链技术在金融科技中的创新应用和风险评估"

    start_time = time.time()

    try:
        # 使用高级优化级别
        result = await run_optimized_research_workflow(
            user_input=user_query,
            workflow_type="analysis",
            optimization_level=WorkflowOptimizationLevel.ADVANCED,
            enable_parallel_tasks=True,
            max_workers=8,
        )

        execution_time = time.time() - start_time

        logger.info(f"高级优化完成，耗时: {execution_time:.2f}秒")
        logger.info(f"优化应用状态: {result.get('optimization_applied', False)}")

        if "workflow_metrics" in result:
            metrics = result["workflow_metrics"]
            logger.info(f"性能指标: {metrics}")

        return result

    except Exception as e:
        logger.error(f"高级优化演示失败: {e}")
        return None


async def demo_parallel_report_generation():
    """演示并行报告生成"""
    logger.info("=== 并行报告生成演示 ===")

    # 定义报告各个部分
    content_sections = [
        "执行摘要",
        "市场分析",
        "技术评估",
        "风险分析",
        "投资建议",
        "结论与展望",
    ]

    start_time = time.time()

    try:
        # 并行生成报告
        result = await run_parallel_report_generation(
            content_sections=content_sections,
            report_type="investment_analysis",
            user_context="针对科技投资者的专业分析报告",
            optimization_level=WorkflowOptimizationLevel.ADVANCED,
        )

        execution_time = time.time() - start_time

        logger.info(f"并行报告生成完成，耗时: {execution_time:.2f}秒")
        logger.info(f"处理部分数量: {result.get('sections_processed', 0)}")

        if "sections" in result:
            logger.info(f"生成的报告部分: {len(result['sections'])}")
            for i, section in enumerate(result["sections"]):
                if isinstance(section, dict) and "content" in section:
                    logger.info(f"  部分 {i+1}: {section['content'][:100]}...")

        return result

    except Exception as e:
        logger.error(f"并行报告生成演示失败: {e}")
        return None


async def demo_maximum_optimization():
    """演示最大优化级别"""
    logger.info("=== 最大优化级别演示 ===")

    user_query = "全面研究量子计算在密码学、机器学习和药物发现领域的突破性应用"

    start_time = time.time()

    try:
        # 使用最大优化级别
        result = await run_optimized_research_workflow(
            user_input=user_query,
            workflow_type="comprehensive_research",
            optimization_level=WorkflowOptimizationLevel.MAXIMUM,
            enable_parallel_tasks=True,
            max_workers=12,
        )

        execution_time = time.time() - start_time

        logger.info(f"最大优化完成，耗时: {execution_time:.2f}秒")
        logger.info(f"优化应用状态: {result.get('optimization_applied', False)}")

        if "workflow_metrics" in result:
            metrics = result["workflow_metrics"]
            logger.info("详细性能指标:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")

        return result

    except Exception as e:
        logger.error(f"最大优化演示失败: {e}")
        return None


async def demo_performance_comparison():
    """演示性能对比"""
    logger.info("=== 性能对比演示 ===")

    user_query = "分析5G技术对物联网发展的影响"

    # 测试标准工作流
    logger.info("测试标准工作流...")
    start_time = time.time()
    try:
        standard_result = await run_agent_workflow_async(
            user_input=user_query, enable_advanced_optimization=False
        )
        standard_time = time.time() - start_time
        logger.info(f"标准工作流耗时: {standard_time:.2f}秒")
    except Exception as e:
        logger.error(f"标准工作流失败: {e}")
        standard_time = None

    # 测试优化工作流
    logger.info("测试优化工作流...")
    start_time = time.time()
    try:
        optimized_result = await run_optimized_research_workflow(
            user_input=user_query,
            workflow_type="research",
            optimization_level=WorkflowOptimizationLevel.ADVANCED,
            enable_parallel_tasks=True,
            max_workers=6,
        )
        optimized_time = time.time() - start_time
        logger.info(f"优化工作流耗时: {optimized_time:.2f}秒")
    except Exception as e:
        logger.error(f"优化工作流失败: {e}")
        optimized_time = None

    # 性能对比
    if standard_time and optimized_time:
        improvement = ((standard_time - optimized_time) / standard_time) * 100
        logger.info(f"性能提升: {improvement:.1f}%")

        if improvement > 0:
            logger.info(f"优化工作流比标准工作流快 {improvement:.1f}%")
        else:
            logger.info(f"标准工作流比优化工作流快 {abs(improvement):.1f}%")

    return {
        "standard_time": standard_time,
        "optimized_time": optimized_time,
        "improvement_percentage": (
            improvement if standard_time and optimized_time else None
        ),
    }


async def demo_error_handling():
    """演示错误处理和回退机制"""
    logger.info("=== 错误处理和回退机制演示 ===")

    # 使用一个可能导致错误的复杂查询
    user_query = "分析一个不存在的技术领域的发展趋势"

    try:
        result = await run_optimized_research_workflow(
            user_input=user_query,
            workflow_type="research",
            optimization_level=WorkflowOptimizationLevel.ADVANCED,
            enable_parallel_tasks=True,
            max_workers=4,
        )

        if result.get("fallback_used", False):
            logger.info("成功使用回退机制处理错误")
            logger.info(f"原始错误: {result.get('original_error', 'Unknown')}")
        else:
            logger.info("查询成功处理，无需回退")

        return result

    except Exception as e:
        logger.error(f"错误处理演示失败: {e}")
        return None


async def main():
    """主演示函数"""
    logger.info("开始 DeerFlow 并行化处理优化演示")

    demos = [
        ("基础优化", demo_basic_optimization),
        ("高级优化", demo_advanced_optimization),
        ("并行报告生成", demo_parallel_report_generation),
        ("最大优化级别", demo_maximum_optimization),
        ("性能对比", demo_performance_comparison),
        ("错误处理", demo_error_handling),
    ]

    results = {}

    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始演示: {demo_name}")
        logger.info(f"{'='*50}")

        try:
            result = await demo_func()
            results[demo_name] = result
            logger.info(f"演示 '{demo_name}' 完成")
        except Exception as e:
            logger.error(f"演示 '{demo_name}' 失败: {e}")
            results[demo_name] = {"error": str(e)}

        # 在演示之间添加短暂延迟
        await asyncio.sleep(1)

    # 总结
    logger.info(f"\n{'='*50}")
    logger.info("演示总结")
    logger.info(f"{'='*50}")

    for demo_name, result in results.items():
        if result and not result.get("error"):
            logger.info(f"✓ {demo_name}: 成功")
        else:
            logger.info(f"✗ {demo_name}: 失败")

    logger.info("\nDeerFlow 并行化处理优化演示完成")
    return results


if __name__ == "__main__":
    # 运行演示
    try:
        results = asyncio.run(main())
        print("\n演示完成！查看日志了解详细信息。")
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示失败: {e}")
