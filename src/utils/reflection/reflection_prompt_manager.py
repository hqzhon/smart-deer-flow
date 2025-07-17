from typing import Dict, Any, Optional
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
        if language == Language.ZH_CN:
            prompt = f"""基于以下研究主题和知识缺口，生成3-5个具体的后续查询问题：

研究主题：{research_topic}

知识缺口：
{chr(10).join(f"- {gap}" for gap in knowledge_gaps)}

优先关注领域：
{chr(10).join(f"- {area}" for area in priority_areas)}

请生成具体、可操作的查询问题，每个问题应该：
1. 针对特定的知识缺口
2. 包含必要的上下文信息
3. 能够通过搜索或研究获得答案
4. 有助于完善整体研究

以JSON数组格式返回查询列表。"""
        else:
            prompt = f"""Based on the following research topic and knowledge gaps, generate 3-5 specific follow-up queries:

Research Topic: {research_topic}

Knowledge Gaps:
{chr(10).join(f"- {gap}" for gap in knowledge_gaps)}

Priority Areas:
{chr(10).join(f"- {area}" for area in priority_areas)}

Generate specific, actionable query questions where each question should:
1. Target a specific knowledge gap
2. Include necessary context information
3. Be answerable through search or research
4. Help complete the overall research

Return the query list in JSON array format."""

        return prompt

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
