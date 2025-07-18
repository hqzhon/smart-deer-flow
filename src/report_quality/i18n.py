# DeerFlow Report Quality - Internationalization Support

from enum import Enum


class Language(Enum):
    """Supported languages"""

    ZH_CN = "zh_cn"  # Simplified Chinese
    EN_US = "en_us"  # English


class I18nManager:
    """Internationalization manager"""

    def __init__(self, default_language: Language = Language.ZH_CN):
        self.current_language = default_language
        self.translations = {}
        self._load_translations()

    def _load_translations(self):
        """Load translation files"""
        # Define translations directly in code for now
        # In production, these could be loaded from JSON files
        self.translations = {
            Language.ZH_CN: {
                # Limitation types
                "limitation_types": {
                    "sample_size": "样本规模",
                    "time_period": "时间范围",
                    "geographic_scope": "地理范围",
                    "methodology": "研究方法",
                    "data_quality": "数据质量",
                    "bias": "偏差",
                    "external_validity": "外部效度",
                    "statistical_power": "统计功效",
                },
                # Confidence levels
                "confidence_levels": {
                    "very_high": "非常高",
                    "high": "高",
                    "medium": "中等",
                    "low": "低",
                    "very_low": "非常低",
                },
                # Evidence quality
                "evidence_quality": {
                    "strong": "强",
                    "moderate": "中等",
                    "weak": "弱",
                    "insufficient": "不足",
                },
                # Impact levels
                "impact_levels": {"high": "高", "medium": "中等", "low": "低"},
                # Limitation descriptions
                "limitation_descriptions": {
                    "small_sample_size": (
                        "样本规模较小（n={sample_size}），可能影响结果的普适性"
                    ),
                    "short_time_span": "分析时间跨度较短，可能无法捕捉长期趋势",
                    "outdated_data": "数据相对过时（{year}年），当前相关性可能有限",
                    "geographic_limitation": (
                        "研究范围限于特定地理区域，结果外推性可能有限"
                    ),
                    "single_methodology": "依赖单一研究方法可能引入方法学偏差",
                    "single_data_source": "依赖单一数据源可能引入数据偏差",
                    "no_control_group": "缺乏对照组使得难以建立因果关系",
                    "convenience_sampling": "便利抽样可能导致样本不具代表性",
                    "temporal_bias": "特殊时期收集的数据可能缺乏普适性",
                },
                # Mitigation suggestions
                "mitigation_suggestions": {
                    "multiple_data_sources": "使用多个数据源进行交叉验证",
                    "assess_source_reliability": "评估数据源可靠性",
                    "explain_single_source_limits": "说明单一数据源的局限性",
                    "setup_control_groups": "设置适当的对照组",
                    "quasi_experimental_design": "使用自然或准实验设计",
                    "clarify_correlation_causation": "明确区分相关性和因果性",
                    "increase_sample_size": "增加样本量",
                    "use_statistical_methods": "使用统计方法评估代表性",
                    "acknowledge_sample_limits": "承认样本局限性",
                    "extend_time_range": "扩展时间范围",
                    "use_recent_data": "使用更新的数据",
                    "explain_time_limits": "说明时间局限性",
                    "explain_short_term_limits": "说明短期数据的局限性和可能的偏差",
                    "compare_historical_trends": "与历史趋势进行对比分析以增强可信度",
                    "expand_geographic_scope": "扩大地理范围",
                    "use_representative_regions": "使用代表性地区",
                    "explain_geographic_limits": "说明地理局限性",
                    "improve_methodology": "改进研究方法",
                    "use_validated_methods": "使用经过验证的方法",
                    "explain_method_limits": "说明方法局限性",
                    "multiple_methods": "采用多种研究方法进行交叉验证",
                    "explain_methodology_limits": (
                        "详细说明所采用方法的局限性和适用范围"
                    ),
                    "sensitivity_analysis": "进行敏感性分析以测试结论的稳健性",
                },
                # Bias types and descriptions
                "bias_types": {
                    "confirmation_bias": "确认偏差：倾向于寻找支持既有观点的证据",
                    "survivorship_bias": "幸存者偏差：只考虑成功案例而忽略失败案例",
                    "availability_bias": "可得性偏差：过度依赖容易获得的信息",
                    "anchoring_bias": "锚定偏差：过度依赖初始信息",
                    "selection_bias": "选择偏差",
                    "temporal_bias": "时间偏差",
                },
                # Bias mitigation measures
                "bias_mitigation": {
                    "seek_opposing_views": "主动寻找反对观点",
                    "multiple_explanations": "考虑多种可能解释",
                    "systematic_evidence_collection": "建立系统性证据收集流程",
                    "include_failure_analysis": "包含失败案例分析",
                    "consider_exited_companies": "考虑退出市场的公司",
                    "analyze_complete_datasets": "分析完整数据集",
                    "systematic_data_collection": "使用系统性数据收集",
                    "avoid_single_source": "避免依赖单一信息源",
                    "consider_hard_to_obtain_info": "考虑难以获得的信息",
                    "multiple_perspectives": "从多个角度评估",
                    "different_reference_points": "使用不同参考点",
                    "conduct_sensitivity_analysis": "进行敏感性分析",
                },
                # Confidence reasoning
                "confidence_reasoning": {
                    "very_high": "基于{count}项高质量证据，结论具有很高可信度",
                    "high": "有{count}项支持证据，结论相对可靠",
                    "medium": "有{count}项证据支持，但需要更多验证",
                    "low": "证据有限（{count}项），结论应谨慎对待",
                    "very_low": "证据不足（{count}项），结论具有较高不确定性",
                    "error": "置信度评估错误",
                },
                # Potential impacts
                "potential_impacts": {
                    "affect_objectivity_accuracy": "可能影响结论的客观性和准确性",
                    "not_generalizable": "结果可能无法推广到总体",
                    "not_apply_normal_periods": "结论可能不适用于正常时期",
                },
                # Section headers
                "section_headers": {
                    "data_limitations": "## 数据局限性\n\n",
                    "potential_bias_alerts": "## 潜在偏差提醒\n\n",
                    "conclusion_confidence": "## 结论置信度\n\n",
                    "statement": "**陈述**",
                    "confidence": "**置信度**",
                    "reasoning": "**推理**",
                    "supporting_evidence": "**支持证据**",
                    "suggested_improvements": "建议改进：\n",
                    "mitigation_measures": "缓解措施：\n",
                    "potential_impact": "潜在影响",
                },
                # Interactive elements
                "interactive_elements": {
                    "chart_title": "图表",
                    "chart_description": "点击查看详细信息",
                    "clickable_chart_title": "可点击图表：{alt_text}",
                    "clickable_chart_description": "点击查看图表详细信息和数据",
                    "table_title": "数据表格",
                    "table_description": "可排序的数据表格",
                    "link_title": "外部链接",
                    "link_description": "点击访问外部资源",
                    "source_link_title": "参考链接：{link_text}",
                    "source_link_description": "点击访问外部参考资源",
                    "code_viewer_title": "代码查看器",
                    "code_viewer_description": "点击查看代码详情",
                    "click_for_details": "🔗 点击查看详情",
                    "data_sources": "数据源",
                    "code_blocks": "代码块",
                    "no_navigation_items": "无导航项目",
                    "no_data_sources": "无数据源信息",
                    "view_source_file": "查看源文件",
                    "no_code_blocks": "无代码块",
                    "language_label": "语言",
                    "view_code_button": "查看代码",
                    "code_block_description": "{language} 代码块 #{index}",
                    "dynamic_table_title": "动态表格 #{index}",
                    "enhanced_interactive_report": "增强交互式报告",
                    "data_source_description": "数据源：{name}",
                    "external_data_source_description": "外部数据源：{name}",
                    "default_report_title": "交互式报告",
                },
            },
            Language.EN_US: {
                # Limitation types
                "limitation_types": {
                    "sample_size": "Sample Size",
                    "time_range": "Time Range",
                    "geographic_scope": "Geographic Scope",
                    "methodology": "Methodology",
                    "data_quality": "Data Quality",
                },
                # Confidence levels
                "confidence_levels": {
                    "very_high": "Very High",
                    "high": "High",
                    "medium": "Medium",
                    "low": "Low",
                    "very_low": "Very Low",
                },
                # Evidence quality
                "evidence_quality": {
                    "strong": "Strong",
                    "moderate": "Moderate",
                    "weak": "Weak",
                    "insufficient": "Insufficient",
                },
                # Impact levels
                "impact_levels": {"high": "High", "medium": "Medium", "low": "Low"},
                # Limitation descriptions
                "limitation_descriptions": {
                    "small_sample": (
                        "Small sample size may affect statistical significance"
                    ),
                    "short_time_range": (
                        "Short analysis period may not capture long-term trends"
                    ),
                    "outdated_data": (
                        "Data may be outdated and not reflect current situation"
                    ),
                    "limited_geographic_scope": (
                        "Analysis limited to specific geographic regions"
                    ),
                    "methodology_constraints": (
                        "Research methodology has certain limitations"
                    ),
                    "single_data_source": (
                        "Reliance on single data source may introduce data bias"
                    ),
                    "no_control_group": (
                        "Lack of control group makes it difficult to establish causal relationships"
                    ),
                },
                # Mitigation suggestions
                "mitigation_suggestions": {
                    "multiple_data_sources": (
                        "Use multiple data sources for cross-validation"
                    ),
                    "assess_source_reliability": "Assess data source reliability",
                    "explain_single_source_limits": (
                        "Explain limitations of single data source"
                    ),
                    "setup_control_groups": "Set up appropriate control groups",
                    "quasi_experimental_design": (
                        "Use natural or quasi-experimental designs"
                    ),
                    "clarify_correlation_causation": (
                        "Clarify distinction between correlation and causation"
                    ),
                    "increase_sample_size": "Increase sample size",
                    "use_statistical_methods": (
                        "Use statistical methods to assess representativeness"
                    ),
                    "acknowledge_sample_limits": "Acknowledge sample limitations",
                    "extend_time_range": "Extend time range",
                    "use_recent_data": "Use more recent data",
                    "explain_time_limits": "Explain temporal limitations",
                    "explain_short_term_limits": (
                        "Explain limitations and potential biases of short-term data"
                    ),
                    "compare_historical_trends": (
                        "Compare with historical trends to enhance credibility"
                    ),
                    "expand_geographic_scope": "Expand geographic scope",
                    "use_representative_regions": "Use representative regions",
                    "explain_geographic_limits": "Explain geographic limitations",
                    "improve_methodology": "Improve research methodology",
                    "use_validated_methods": "Use validated methods",
                    "explain_method_limits": "Explain methodological limitations",
                    "multiple_methods": (
                        "Employ multiple research methods for cross-validation"
                    ),
                    "explain_methodology_limits": (
                        "Provide detailed explanation of methodological limitations and scope"
                    ),
                    "sensitivity_analysis": (
                        "Conduct sensitivity analysis to test robustness of conclusions"
                    ),
                },
                "confidence_reasoning": {
                    "very_high": (
                        "Based on {count} high-quality evidence items, conclusion has very high confidence"
                    ),
                    "high": (
                        "With {count} supporting evidence items, conclusion is relatively reliable"
                    ),
                    "medium": (
                        "With {count} evidence items supporting, but more verification needed"
                    ),
                    "low": (
                        "Limited evidence ({count} items), conclusion should be treated with caution"
                    ),
                    "very_low": (
                        "Insufficient evidence ({count} items), conclusion has high uncertainty"
                    ),
                    "error": "Confidence assessment error",
                    "high_indicator": "High confidence indicator found: '{indicator}'",
                    "moderate_indicator": (
                        "Moderate confidence indicator found: '{indicator}'"
                    ),
                    "neutral_indicator": (
                        "Neutral confidence indicator found: '{indicator}'"
                    ),
                    "low_indicator": "Low confidence indicator found: '{indicator}'",
                    "very_low_indicator": (
                        "Very low confidence indicator found: '{indicator}'"
                    ),
                    "evidence_count": "Supporting evidence count: {count}",
                    "no_evidence": "No supporting evidence provided",
                },
                "bias_types": {
                    "confirmation_bias": (
                        "Confirmation bias: tendency to seek evidence supporting existing views"
                    ),
                    "survivorship_bias": (
                        "Survivorship bias: only considering successful cases while ignoring failures"
                    ),
                    "availability_bias": (
                        "Availability bias: over-reliance on easily accessible information"
                    ),
                    "anchoring_bias": (
                        "Anchoring bias: over-reliance on initial information"
                    ),
                    "selection_bias": (
                        "Selection bias: convenience sampling may lead to non-representative samples"
                    ),
                    "temporal_bias": (
                        "Temporal bias: data collected during special periods may lack generalizability"
                    ),
                },
                "potential_impacts": {
                    "confirmation_bias": (
                        "May affect objectivity and accuracy of conclusions"
                    ),
                    "survivorship_bias": "May lead to overly optimistic conclusions",
                    "availability_bias": (
                        "May lead to decisions based on incomplete information"
                    ),
                    "anchoring_bias": "May limit consideration of alternative options",
                    "selection_bias": (
                        "Results may not be generalizable to the population"
                    ),
                    "temporal_bias": "Conclusions may not apply to normal periods",
                },
                "bias_mitigation": {
                    "confirmation_bias": [
                        "Actively seek opposing viewpoints",
                        "Consider multiple possible explanations",
                        "Establish systematic evidence collection process",
                    ],
                    "survivorship_bias": [
                        "Include failure case analysis",
                        "Consider companies that exited the market",
                        "Analyze complete datasets",
                    ],
                    "availability_bias": [
                        "Use systematic data collection",
                        "Avoid reliance on single information source",
                        "Consider hard-to-obtain information",
                    ],
                    "anchoring_bias": [
                        "Evaluate from multiple perspectives",
                        "Use different reference points",
                        "Conduct sensitivity analysis",
                    ],
                    "selection_bias": [
                        "Use random sampling",
                        "Ensure sample representativeness",
                        "Analyze sample characteristics",
                    ],
                    "temporal_bias": [
                        "Include data from multiple periods",
                        "Analyze impact of period characteristics",
                        "Explain temporal limitations",
                    ],
                },
                "section_titles": {
                    "data_limitations": "Data Limitations",
                    "bias_alerts": "Potential Bias Alerts",
                    "conclusion_confidence": "Conclusion Confidence",
                    "suggested_improvements": "Suggested improvements",
                    "potential_impact": "Potential impact",
                    "mitigation_measures": "Mitigation measures",
                    "statement": "Statement",
                    "confidence": "Confidence",
                    "reasoning": "Reasoning",
                    "supporting_evidence": "Supporting evidence",
                },
                # Interactive elements
                "interactive_elements": {
                    "chart_title": "Chart",
                    "chart_description": "Click for details",
                    "clickable_chart_title": "Clickable Chart: {alt_text}",
                    "clickable_chart_description": (
                        "Click to view detailed chart information and data"
                    ),
                    "table_title": "Data Table",
                    "table_description": "Sortable data table",
                    "link_title": "External Link",
                    "link_description": "Click to visit external resource",
                    "source_link_title": "Source Link: {link_text}",
                    "source_link_description": "Click to visit external source",
                    "code_viewer_title": "Code Viewer",
                    "code_viewer_description": "Click to view code details",
                    "click_for_details": "🔗 Click for details",
                    "data_sources": "Data Sources",
                    "code_blocks": "Code Blocks",
                    "no_navigation_items": "No navigation items",
                    "no_data_sources": "No data source information",
                    "view_source_file": "View source file",
                    "no_code_blocks": "No code blocks",
                    "language_label": "Language",
                    "view_code_button": "View Code",
                    "code_block_description": "{language} Code Block #{index}",
                    "dynamic_table_title": "Dynamic Table #{index}",
                    "enhanced_interactive_report": "Enhanced Interactive Report",
                    "data_source_description": "Data source: {name}",
                    "external_data_source_description": "External data source: {name}",
                    "default_report_title": "Interactive Report",
                },
            },
        }

    def set_language(self, language: Language):
        """Set current language"""
        self.current_language = language

    def get_text(self, category: str, key: str, **kwargs) -> str:
        """Get translated text"""
        try:
            text = self.translations[self.current_language][category][key]
            if kwargs:
                return text.format(**kwargs)
            return text
        except KeyError:
            # Fallback to English if translation not found
            try:
                text = self.translations[Language.EN_US][category][key]
                if kwargs:
                    return text.format(**kwargs)
                return text
            except KeyError:
                return f"[Missing translation: {category}.{key}]"

    def get_limitation_type_text(self, limitation_type: str) -> str:
        """Get limitation type text"""
        return self.get_text("limitation_types", limitation_type)

    def get_confidence_level_text(self, confidence_level: str) -> str:
        """Get confidence level text"""
        return self.get_text("confidence_levels", confidence_level)

    def get_impact_level_text(self, impact_level: str) -> str:
        """Get impact level text"""
        return self.get_text("impact_levels", impact_level)


# Global instance
_i18n_manager = I18nManager()


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance"""
    return _i18n_manager


def set_language(language: Language):
    """Set global language"""
    _i18n_manager.set_language(language)


def get_text(language: Language, category: str, key: str, **kwargs) -> str:
    """Get translated text using specified language"""
    try:
        text = _i18n_manager.translations[language][category][key]
        if kwargs:
            return text.format(**kwargs)
        return text
    except KeyError:
        # Fallback to English if translation not found
        try:
            text = _i18n_manager.translations[Language.EN_US][category][key]
            if kwargs:
                return text.format(**kwargs)
            return text
        except KeyError:
            return f"[Missing translation: {category}.{key}]"
