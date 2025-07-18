from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from ..template import apply_prompt_template
from ...report_quality.i18n import Language, I18nManager


class ReflectionPromptManager:
    """管理反射机制的多语言Prompt模板"""

    def __init__(self, i18n_manager: Optional[I18nManager] = None):
        self.i18n_manager = i18n_manager or I18nManager()
        self.template_dir = Path(__file__).parent

    def get_reflection_analysis_prompt(
        self,
        research_topic: str,
        current_step_index: int,
        total_steps: int,
        current_reflection_loop: int,
        max_reflection_loops: int,
        steps_summary: str,
        results_summary: str,
        observations_summary: str,
        language: Language = Language.EN_US,
    ) -> str:
        """获取反射分析的Prompt

        Args:
            research_topic: 研究主题
            current_step_index: 当前步骤索引
            total_steps: 总步骤数
            current_reflection_loop: 当前反射循环
            max_reflection_loops: 最大反射循环数
            steps_summary: 已完成步骤摘要
            results_summary: 执行结果摘要
            observations_summary: 当前观察摘要
            language: 目标语言

        Returns:
            格式化的反射分析Prompt
        """
        # 设置语言环境
        locale = "zh-CN" if language == Language.ZH_CN else "en-US"

        # 准备模板变量
        template_vars = {
            "CURRENT_TIME": datetime.now().isoformat(),
            "locale": locale,
            "research_topic": research_topic,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_step_index": current_step_index,
            "total_steps": total_steps,
            "current_reflection_loop": current_reflection_loop,
            "max_reflection_loops": max_reflection_loops,
            "steps_summary": steps_summary,
            "results_summary": results_summary,
            "observations_summary": observations_summary,
        }

        # Apply template using the project's template system
        # Convert template_vars to AgentState format
        agent_state = {**template_vars, "messages": []}
        result = apply_prompt_template("reflection_analysis", agent_state)
        # Extract the system prompt content
        if result and len(result) > 0 and "content" in result[0]:
            return result[0]["content"]
        return "Template rendering failed"

    def get_follow_up_queries_prompt(
        self,
        research_topic: str,
        knowledge_gaps: list,
        priority_areas: list,
        language: Language = Language.EN_US,
    ) -> str:
        """获取后续查询生成的Prompt

        Args:
            research_topic: 研究主题
            knowledge_gaps: 知识缺口列表
            priority_areas: 优先关注领域
            language: 目标语言

        Returns:
            格式化的后续查询生成Prompt
        """
        # 设置语言环境
        locale = "zh-CN" if language == Language.ZH_CN else "en-US"

        # 准备模板变量
        template_vars = {
            "CURRENT_TIME": datetime.now().isoformat(),
            "locale": locale,
            "research_topic": research_topic,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "knowledge_gaps": knowledge_gaps,
            "priority_areas": priority_areas,
        }

        # Apply template using the project's template system
        # Convert template_vars to AgentState format
        agent_state = {**template_vars, "messages": []}
        result = apply_prompt_template("reflection_follow_up_queries", agent_state)
        # Extract the system prompt content
        if result and len(result) > 0 and "content" in result[0]:
            return result[0]["content"]
        return "Template rendering failed"

    def get_quality_assessment_labels(
        self, language: Language = Language.EN_US
    ) -> Dict[str, str]:
        """获取质量评估的标签文本

        Args:
            language: 目标语言

        Returns:
            质量评估标签字典
        """
        if language == Language.ZH_CN:
            return {
                "completeness": "完整性",
                "accuracy": "准确性",
                "depth": "深度",
                "relevance": "相关性",
                "currency": "时效性",
                "overall_score": "总体评分",
            }
        else:
            return {
                "completeness": "Completeness",
                "accuracy": "Accuracy",
                "depth": "Depth",
                "relevance": "Relevance",
                "currency": "Currency",
                "overall_score": "Overall Score",
            }

    def get_recommendation_categories(
        self, language: Language = Language.EN_US
    ) -> Dict[str, str]:
        """获取建议分类的标签文本

        Args:
            language: 目标语言

        Returns:
            建议分类标签字典
        """
        if language == Language.ZH_CN:
            return {
                "methodology": "方法论改进",
                "sources": "信息源扩展",
                "analysis": "分析深化",
                "validation": "结果验证",
                "presentation": "呈现优化",
            }
        else:
            return {
                "methodology": "Methodology Improvement",
                "sources": "Source Expansion",
                "analysis": "Analysis Deepening",
                "validation": "Result Validation",
                "presentation": "Presentation Optimization",
            }
