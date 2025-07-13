#!/usr/bin/env python3
"""
Researcher Phase 4 使用示例

这个示例展示了如何使用 Phase 4 的所有主要功能：
1. 智能渐进式启用
2. 高级监控与预测分析
3. 智能配置优化
4. 系统集成与协调
"""

import time
import json
from datetime import datetime
from pathlib import Path

# 导入 Phase 4 组件
from utils.researcher_progressive_enablement_phase4 import get_advanced_progressive_enabler
from utils.researcher_isolation_metrics_phase4 import get_advanced_isolation_metrics
from utils.researcher_config_optimizer_phase4 import get_config_optimizer, ConfigOptimizationLevel
from utils.researcher_phase4_integration import get_phase4_system


def demo_advanced_progressive_enablement():
    """演示智能渐进式启用功能"""
    print("\n=== 智能渐进式启用演示 ===")
    
    enabler = get_advanced_progressive_enabler()
    
    # 测试不同复杂度的场景
    scenarios = [
        {
            "name": "简单查询",
            "context": {
                "task_description": "查询用户基本信息",
                "expected_steps": 2,
                "context_size": 500,
                "has_external_deps": False,
                "user_priority": "low"
            }
        },
        {
            "name": "复杂分析",
            "context": {
                "task_description": "分析多维度数据关联性并生成报告",
                "expected_steps": 12,
                "context_size": 25000,
                "has_external_deps": True,
                "user_priority": "high"
            }
        },
        {
            "name": "中等任务",
            "context": {
                "task_description": "处理文档并提取关键信息",
                "expected_steps": 6,
                "context_size": 8000,
                "has_external_deps": False,
                "user_priority": "medium"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        
        # 获取决策和解释
        should_isolate, explanation = enabler.should_enable_isolation_with_explanation(
            scenario['context']
        )
        
        print(f"  是否启用隔离: {should_isolate}")
        print(f"  决策理由: {explanation['reason']}")
        print(f"  置信度: {explanation['confidence']:.1%}")
        print(f"  隔离分数: {explanation['isolation_score']:.2f}")
        
        # 模拟执行结果并记录
        outcome = {
            "isolation_used": should_isolate,
            "success": True,
            "token_savings": 0.3 if should_isolate else 0.0,
            "performance_overhead": 0.1 if should_isolate else 0.0,
            "execution_time": 60 + (scenario['context']['expected_steps'] * 10)
        }
        
        enabler.record_scenario_outcome(scenario['context'], outcome)
        print(f"  已记录执行结果: 成功={outcome['success']}, Token节省={outcome['token_savings']:.1%}")
    
    # 显示统计信息
    stats = enabler.get_enablement_statistics()
    print(f"\n统计信息:")
    print(f"  总决策次数: {stats['total_decisions']}")
    print(f"  启用隔离次数: {stats['isolation_enabled']}")
    print(f"  平均Token节省: {stats['avg_token_savings']:.1%}")
    print(f"  平均性能开销: {stats['avg_performance_overhead']:.1%}")
    
    # 执行自动调优
    print("\n执行自动调优...")
    tuning_result = enabler.auto_tune_parameters()
    if tuning_result['parameters_updated']:
        print(f"  调优完成，更新了 {len(tuning_result['updated_parameters'])} 个参数")
        for param, value in tuning_result['updated_parameters'].items():
            print(f"    {param}: {value}")
    else:
        print("  当前参数已是最优，无需调整")


def demo_advanced_metrics():
    """演示高级监控与预测分析功能"""
    print("\n=== 高级监控与预测分析演示 ===")
    
    metrics = get_advanced_isolation_metrics()
    
    # 模拟一些会话数据
    print("\n模拟隔离会话...")
    for i in range(3):
        session_id = metrics.start_isolation_session({
            "task_type": f"analysis_{i}",
            "context_size": 5000 + i * 2000,
            "expected_complexity": "medium"
        })
        
        # 模拟会话进行
        time.sleep(0.1)  # 模拟处理时间
        
        metrics.update_isolation_session(session_id, {
            "tokens_saved": 1000 + i * 500,
            "compression_ratio": 0.7 + i * 0.1,
            "performance_impact": 0.1 + i * 0.05
        })
        
        metrics.end_isolation_session(session_id, {
            "success": True,
            "final_tokens_saved": 1200 + i * 600,
            "total_execution_time": 30 + i * 10
        })
        
        print(f"  会话 {i+1} 完成: 节省 {1200 + i * 600} tokens")
    
    # 获取实时仪表板
    print("\n实时仪表板数据:")
    dashboard = metrics.get_real_time_dashboard()
    current = dashboard['current_metrics']
    print(f"  系统健康分数: {current['system_health_score']:.1f}")
    print(f"  活跃会话数: {current['active_sessions']}")
    print(f"  1小时成功率: {current['success_rate_1h']:.1%}")
    print(f"  平均压缩比: {current['avg_compression_ratio']:.2f}")
    print(f"  系统状态: {dashboard['system_status']}")
    
    # 获取预测洞察
    print("\n预测分析洞察:")
    insights = metrics.get_predictive_insights()
    print(f"  性能趋势: {insights['performance_trend']['direction']} (强度: {insights['performance_trend']['strength']:.2f})")
    print(f"  7天预测成功率: {insights['predictions']['success_rate_7d']:.1%}")
    print(f"  预测置信度: {insights['predictions']['confidence']:.1%}")
    
    print("  优化建议:")
    for rec in insights['recommendations']:
        print(f"    - {rec}")
    
    # 设置告警回调
    def alert_handler(alert):
        print(f"  📢 告警: {alert.title} ({alert.level.value})")
        print(f"     描述: {alert.description}")
    
    metrics.add_alert_callback(alert_handler)
    print("\n已设置告警回调，监控系统状态...")
    
    # 导出详细报告
    report_file = metrics.export_advanced_report()
    print(f"\n详细监控报告已导出到: {report_file}")


def demo_config_optimization():
    """演示智能配置优化功能"""
    print("\n=== 智能配置优化演示 ===")
    
    optimizer = get_config_optimizer()
    
    # 获取当前配置状态
    print("\n当前配置状态:")
    report = optimizer.get_configuration_report()
    print(f"  配置健康分数: {report['config_health_score']:.1f}")
    print(f"  自动调优状态: {report['auto_tuning_status']}")
    print(f"  当前档案: {report['current_profile']}")
    
    # 测试不同的配置档案
    profiles = ["balanced", "high_performance", "low_latency", "memory_efficient"]
    
    print("\n测试不同配置档案:")
    for profile in profiles:
        print(f"\n应用 {profile} 档案...")
        result = optimizer.apply_profile(profile)
        if result['success']:
            print(f"  ✅ 成功应用，更改了 {result['changes_applied']} 项配置")
            for change in result['applied_changes']:
                print(f"    {change['parameter']}: {change['old_value']} -> {change['new_value']}")
        else:
            print(f"  ❌ 应用失败: {result['error']}")
    
    # 模拟性能数据并生成建议
    print("\n基于性能数据生成配置建议:")
    performance_data = {
        "success_rate_1h": 0.75,  # 成功率偏低
        "performance_overhead_1h": 0.35,  # 性能开销较高
        "avg_compression_ratio_1h": 0.85,  # 压缩比良好
        "resource_utilization": 0.8  # 资源利用率较高
    }
    
    recommendations = optimizer.analyze_performance_data(performance_data)
    print(f"  生成了 {len(recommendations)} 项建议:")
    
    for rec in recommendations:
        print(f"    参数: {rec.parameter.value}")
        print(f"    建议值: {rec.recommended_value}")
        print(f"    理由: {rec.reason}")
        print(f"    置信度: {rec.confidence:.1%}")
        print()
    
    # 应用建议
    if recommendations:
        print("应用配置建议...")
        applied = optimizer.apply_recommendations(recommendations, auto_apply=True)
        print(f"  成功应用 {len(applied)} 项更改")
    
    # 启用自动调优
    print("\n启用自动调优 (平衡模式)...")
    optimizer.enable_auto_tuning(ConfigOptimizationLevel.BALANCED)
    
    # 获取更新后的配置报告
    updated_report = optimizer.get_configuration_report()
    print(f"  更新后配置健康分数: {updated_report['config_health_score']:.1f}")
    print(f"  自动调优状态: {updated_report['auto_tuning_status']}")


def demo_system_integration():
    """演示系统集成与协调功能"""
    print("\n=== 系统集成与协调演示 ===")
    
    # 初始化 Phase 4 系统
    print("\n初始化 Phase 4 系统...")
    system = get_phase4_system(enable_auto_optimization=True)
    
    # 获取系统状态
    print("\n系统状态:")
    status = system.get_system_status()
    print(f"  运行模式: {status.mode.value}")
    print(f"  健康分数: {status.health_score:.1f}")
    print(f"  优化分数: {status.optimization_score:.1f}")
    print(f"  性能趋势: {status.performance_trend}")
    print(f"  运行时间: {status.uptime_hours:.1f} 小时")
    print(f"  活跃告警: {status.alerts_count}")
    
    # 设置状态变化回调
    def status_change_handler(new_status):
        print(f"  🔄 系统状态变化: {new_status.mode.value}, 健康分数: {new_status.health_score:.1f}")
    
    system.add_status_change_callback(status_change_handler)
    print("\n已设置状态变化监听器")
    
    # 获取综合报告
    print("\n生成综合系统报告...")
    report = system.get_comprehensive_report()
    
    print("系统概览:")
    sys_status = report['system_status']
    print(f"  模式: {sys_status['mode']}")
    print(f"  健康分数: {sys_status['health_score']:.1f}")
    print(f"  运行时间: {sys_status['uptime_hours']:.1f} 小时")
    print(f"  告警数量: {sys_status['alerts_count']}")
    
    print("\n渐进式启用状态:")
    enabler_status = report['progressive_enabler']
    print(f"  总决策: {enabler_status['total_decisions']}")
    print(f"  启用率: {enabler_status['enablement_rate']:.1%}")
    print(f"  平均Token节省: {enabler_status['avg_token_savings']:.1%}")
    
    print("\n监控指标:")
    metrics_status = report['metrics']
    print(f"  活跃会话: {metrics_status['active_sessions']}")
    print(f"  系统健康: {metrics_status['system_health_score']:.1f}")
    print(f"  成功率: {metrics_status['success_rate']:.1%}")
    
    print("\n配置状态:")
    config_status = report['configuration']
    print(f"  健康分数: {config_status['health_score']:.1f}")
    print(f"  当前档案: {config_status['current_profile']}")
    print(f"  自动调优: {config_status['auto_tuning_enabled']}")
    
    # 测试自动优化
    print("\n执行强制优化...")
    optimization_result = system.force_optimization()
    
    if optimization_result.get('optimization_performed'):
        print("  ✅ 优化执行成功")
        if optimization_result.get('auto_tuning_applied'):
            print(f"    应用了 {optimization_result['applied_count']} 项自动调优")
        if optimization_result.get('config_optimized'):
            print(f"    配置优化: {optimization_result['config_changes']} 项更改")
        if optimization_result.get('thresholds_updated'):
            print(f"    阈值更新: {optimization_result['threshold_changes']} 项调整")
    else:
        print("  ℹ️ 系统已处于最优状态，无需优化")
    
    # 导出系统备份
    print("\n导出系统备份...")
    backup_file = system.export_system_backup()
    print(f"  系统备份已保存到: {backup_file}")
    
    # 显示备份内容概览
    try:
        with open(backup_file, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        print("\n备份内容概览:")
        print(f"  备份时间: {backup_data['metadata']['timestamp']}")
        print(f"  系统版本: {backup_data['metadata']['version']}")
        print(f"  包含组件: {', '.join(backup_data['metadata']['components'])}")
        
        # 显示各组件的数据量
        for component, data in backup_data['data'].items():
            if isinstance(data, dict):
                print(f"  {component}: {len(data)} 项配置")
            elif isinstance(data, list):
                print(f"  {component}: {len(data)} 条记录")
    
    except Exception as e:
        print(f"  备份文件读取失败: {e}")


def main():
    """主演示函数"""
    print("🚀 Researcher Phase 4 功能演示")
    print("=" * 50)
    
    try:
        # 1. 智能渐进式启用演示
        demo_advanced_progressive_enablement()
        
        # 2. 高级监控与预测分析演示
        demo_advanced_metrics()
        
        # 3. 智能配置优化演示
        demo_config_optimization()
        
        # 4. 系统集成与协调演示
        demo_system_integration()
        
        print("\n" + "=" * 50)
        print("✅ Phase 4 功能演示完成！")
        print("\n主要特性:")
        print("  ✓ 智能决策与自动调优")
        print("  ✓ 实时监控与预测分析")
        print("  ✓ 配置优化与档案管理")
        print("  ✓ 系统协调与状态管理")
        print("\n系统已准备就绪，可以开始使用！")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查系统配置和依赖项")
        raise


if __name__ == "__main__":
    main()