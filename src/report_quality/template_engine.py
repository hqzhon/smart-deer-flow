# DeerFlow Report Quality Optimization - Domain-Specific Structured Template Engine

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportDomain(Enum):
    """Report Domain"""

    FINANCIAL = "financial"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    MARKET_RESEARCH = "market_research"
    LEGAL = "legal"
    CONSULTING = "consulting"


class SectionType(Enum):
    """Section Type"""

    EXECUTIVE_SUMMARY = "executive_summary"
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    RECOMMENDATIONS = "recommendations"
    REFERENCES = "references"
    APPENDIX = "appendix"
    RISK_ANALYSIS = "risk_analysis"
    FINANCIAL_METRICS = "financial_metrics"


class ContentType(Enum):
    """Content Type"""

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    LIST = "list"
    FORMULA = "formula"
    CODE = "code"
    QUOTE = "quote"


@dataclass
class ContentBlock:
    """Content Block"""

    block_id: str
    content_type: ContentType
    title: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class ReportSection:
    """Report Section"""

    section_id: str
    section_type: SectionType
    title: str
    content_blocks: List[ContentBlock] = field(default_factory=list)
    required: bool = True
    word_limit: Optional[int] = None
    style_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportTemplate:
    """Report Template"""

    template_id: str
    domain: ReportDomain
    name: str
    description: str
    sections: List[ReportSection] = field(default_factory=list)
    global_style: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


class TemplateValidator:
    """Template Validator"""

    def __init__(self):
        self.validation_rules = {
            "word_count": self._validate_word_count,
            "citation_format": self._validate_citation_format,
            "table_structure": self._validate_table_structure,
            "section_completeness": self._validate_section_completeness,
            "financial_metrics": self._validate_financial_metrics,
            "academic_format": self._validate_academic_format,
        }

    def validate_content_block(
        self, block: ContentBlock, rules: List[str]
    ) -> Dict[str, Any]:
        """Validate content block"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": [],
        }

        for rule in rules:
            if rule in self.validation_rules:
                try:
                    result = self.validation_rules[rule](block)
                    if not result["valid"]:
                        validation_result["is_valid"] = False
                        validation_result["errors"].extend(result.get("errors", []))
                    validation_result["warnings"].extend(result.get("warnings", []))
                    validation_result["suggestions"].extend(
                        result.get("suggestions", [])
                    )
                except Exception as e:
                    logger.error(f"Validation rule {rule} execution failed: {e}")
                    validation_result["errors"].append(
                        f"Validation rule exception: {rule}"
                    )

        return validation_result

    def _validate_word_count(self, block: ContentBlock) -> Dict[str, Any]:
        """Validate word count"""
        if block.content_type != ContentType.TEXT:
            return {"valid": True}

        content = str(block.content)
        word_count = len(content.split())

        errors = []
        warnings = []

        # Check minimum word count
        min_words = block.metadata.get("min_words", 0)
        if word_count < min_words:
            errors.append(f"Insufficient word count: {word_count}/{min_words}")

        # Check maximum word count
        max_words = block.metadata.get("max_words", float("inf"))
        if word_count > max_words:
            warnings.append(f"Word count exceeded: {word_count}/{max_words}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "word_count": word_count,
        }

    def _validate_citation_format(self, block: ContentBlock) -> Dict[str, Any]:
        """Validate citation format"""
        content = str(block.content)
        errors = []
        suggestions = []
        warnings = []

        # Check citation format
        citation_patterns = [
            r"\[([^\]]+)\]\(([^)]+)\)",  # Markdown link format
            r"\([^)]+,\s*\d{4}\)",  # APA format
            r"\[\d+\]",  # Numeric citation format
        ]

        has_citations = any(
            re.search(pattern, content) for pattern in citation_patterns
        )

        if block.metadata.get("requires_citations", False) and not has_citations:
            errors.append("Missing required citations")
            suggestions.append("Please add relevant citations to support your points")

        # Check link validity
        markdown_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        for title, url in markdown_links:
            if not url.startswith(("http://", "https://", "ftp://")):
                warnings.append(f"Possibly invalid link: {url}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "citation_count": len(markdown_links),
        }

    def _validate_table_structure(self, block: ContentBlock) -> Dict[str, Any]:
        """Validate table structure"""
        if block.content_type != ContentType.TABLE:
            return {"valid": True}

        errors = []
        suggestions = []

        if isinstance(block.content, str):
            # Markdown table validation
            lines = block.content.strip().split("\n")
            if len(lines) < 3:
                errors.append(
                    "Table requires at least header row, separator row, and data row"
                )
            else:
                # Check table format
                header_cols = len(lines[0].split("|"))
                separator_cols = len(lines[1].split("|"))

                if header_cols != separator_cols:
                    errors.append(
                        "Table header and separator row column count mismatch"
                    )

                # Check data rows
                for i, line in enumerate(lines[2:], start=2):
                    data_cols = len(line.split("|"))
                    if data_cols != header_cols:
                        errors.append(
                            f"Row {i+1} column count does not match header row"
                        )

        elif isinstance(block.content, dict):
            # JSON table validation
            if "headers" not in block.content or "rows" not in block.content:
                errors.append("JSON table must contain headers and rows fields")
            else:
                header_count = len(block.content["headers"])
                for i, row in enumerate(block.content["rows"]):
                    if len(row) != header_count:
                        errors.append(
                            f"Row {i+1} data column count does not match header"
                        )

        return {"valid": len(errors) == 0, "errors": errors, "suggestions": suggestions}

    def _validate_section_completeness(self, block: ContentBlock) -> Dict[str, Any]:
        """Validate section completeness"""
        errors = []
        warnings = []

        required_elements = block.metadata.get("required_elements", [])
        content = str(block.content).lower()

        for element in required_elements:
            if element.lower() not in content:
                errors.append(f"Missing required element: {element}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_financial_metrics(self, block: ContentBlock) -> Dict[str, Any]:
        """Validate financial metrics"""
        errors = []
        suggestions = []
        warnings = []

        content = str(block.content)

        # Check common financial metrics
        financial_terms = [
            "revenue",
            "profit",
            "ebitda",
            "margin",
            "ratio",
            "growth",
            "income",
            "profit",
            "gross margin",
            "net margin",
            "year-over-year",
            "month-over-month",
        ]

        has_financial_terms = any(term in content.lower() for term in financial_terms)

        if block.metadata.get("domain") == "financial" and not has_financial_terms:
            warnings.append("Recommend adding specific financial metric data")

        # Check number format
        number_pattern = r"[\d,]+\.?\d*%?"
        numbers = re.findall(number_pattern, content)

        if len(numbers) == 0 and block.metadata.get("requires_numbers", False):
            errors.append("Financial analysis requires specific data")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "number_count": len(numbers),
        }

    def _validate_academic_format(self, block: ContentBlock) -> Dict[str, Any]:
        """Validate academic format"""
        errors = []
        suggestions = []

        content = str(block.content)

        # Check academic writing standards
        if block.metadata.get("domain") == "academic":
            # Check for first person usage
            first_person_words = ["I", "we", "i ", "we ", "my ", "our "]
            if any(word in content.lower() for word in first_person_words):
                suggestions.append("Academic writing should avoid first person")

            # Check for assumptions or limitations
            if (
                "assumption" not in content.lower()
                and "limitation" not in content.lower()
            ):
                suggestions.append(
                    "Recommend stating research assumptions or limitations"
                )

        return {"valid": len(errors) == 0, "errors": errors, "suggestions": suggestions}


class ReportBuilder:
    """Report Builder"""

    def __init__(self):
        self.validator = TemplateValidator()
        self.templates: Dict[str, ReportTemplate] = {}
        self.initialize_default_templates()

    def initialize_default_templates(self):
        """Initialize default templates"""
        # Financial analysis report template
        financial_template = ReportTemplate(
            template_id="financial_analysis",
            domain=ReportDomain.FINANCIAL,
            name="Financial Analysis Report",
            description="Standard financial analysis report template",
            global_style={
                "font": "Arial",
                "font_size": 12,
                "line_spacing": 1.5,
                "citation_style": "APA",
            },
        )

        # Add financial report sections
        financial_sections = [
            ReportSection(
                section_id="exec_summary",
                section_type=SectionType.EXECUTIVE_SUMMARY,
                title="Executive Summary",
                required=True,
                word_limit=500,
                style_requirements={"format": "bullet_points"},
            ),
            ReportSection(
                section_id="financial_overview",
                section_type=SectionType.ANALYSIS,
                title="Financial Overview",
                required=True,
                style_requirements={"requires_tables": True, "requires_charts": True},
            ),
            ReportSection(
                section_id="performance_analysis",
                section_type=SectionType.ANALYSIS,
                title="Performance Analysis",
                required=True,
            ),
            ReportSection(
                section_id="risk_assessment",
                section_type=SectionType.RISK_ANALYSIS,
                title="Risk Assessment",
                required=True,
            ),
            ReportSection(
                section_id="recommendations",
                section_type=SectionType.RECOMMENDATIONS,
                title="Investment Recommendations",
                required=True,
            ),
        ]

        financial_template.sections = financial_sections
        self.templates["financial_analysis"] = financial_template

        # Academic research report template
        academic_template = ReportTemplate(
            template_id="academic_research",
            domain=ReportDomain.ACADEMIC,
            name="Academic Research Report",
            description="Standard academic research report template",
        )

        academic_sections = [
            ReportSection(
                section_id="abstract",
                section_type=SectionType.EXECUTIVE_SUMMARY,
                title="Abstract",
                required=True,
                word_limit=300,
            ),
            ReportSection(
                section_id="introduction",
                section_type=SectionType.BACKGROUND,
                title="Introduction",
                required=True,
            ),
            ReportSection(
                section_id="literature_review",
                section_type=SectionType.BACKGROUND,
                title="Literature Review",
                required=True,
            ),
            ReportSection(
                section_id="methodology",
                section_type=SectionType.METHODOLOGY,
                title="Methodology",
                required=True,
            ),
            ReportSection(
                section_id="results",
                section_type=SectionType.RESULTS,
                title="Results",
                required=True,
            ),
            ReportSection(
                section_id="discussion",
                section_type=SectionType.DISCUSSION,
                title="Discussion",
                required=True,
            ),
            ReportSection(
                section_id="conclusion",
                section_type=SectionType.CONCLUSION,
                title="Conclusion",
                required=True,
            ),
            ReportSection(
                section_id="references",
                section_type=SectionType.REFERENCES,
                title="References",
                required=True,
            ),
        ]

        academic_template.sections = academic_sections
        self.templates["academic_research"] = academic_template

        # Market research report template
        market_template = ReportTemplate(
            template_id="market_research",
            domain=ReportDomain.MARKET_RESEARCH,
            name="Market Research Report",
            description="Market research analysis report template",
        )

        market_sections = [
            ReportSection(
                section_id="executive_summary",
                section_type=SectionType.EXECUTIVE_SUMMARY,
                title="Executive Summary",
                required=True,
            ),
            ReportSection(
                section_id="market_overview",
                section_type=SectionType.BACKGROUND,
                title="Market Overview",
                required=True,
            ),
            ReportSection(
                section_id="competitive_analysis",
                section_type=SectionType.ANALYSIS,
                title="Competitive Analysis",
                required=True,
                style_requirements={"requires_tables": True},
            ),
            ReportSection(
                section_id="consumer_insights",
                section_type=SectionType.ANALYSIS,
                title="Consumer Insights",
                required=True,
            ),
            ReportSection(
                section_id="market_trends",
                section_type=SectionType.ANALYSIS,
                title="Market Trends",
                required=True,
            ),
            ReportSection(
                section_id="opportunities",
                section_type=SectionType.RECOMMENDATIONS,
                title="Market Opportunities",
                required=True,
            ),
        ]

        market_template.sections = market_sections
        self.templates["market_research"] = market_template

    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get template"""
        return self.templates.get(template_id)

    def get_templates_by_domain(self, domain: ReportDomain) -> List[ReportTemplate]:
        """Get templates by domain"""
        return [
            template
            for template in self.templates.values()
            if template.domain == domain
        ]

    def validate_report_content(
        self, template_id: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate report content"""
        template = self.get_template(template_id)
        if not template:
            return {"error": f"Template not found: {template_id}"}

        validation_results = {}
        overall_valid = True

        # Validate each section
        for section in template.sections:
            section_content = content.get(section.section_id)

            if section.required and not section_content:
                validation_results[section.section_id] = {
                    "is_valid": False,
                    "errors": ["Required section missing"],
                    "warnings": [],
                    "suggestions": [],
                }
                overall_valid = False
                continue

            if section_content:
                # Validate section content
                section_result = self._validate_section_content(
                    section, section_content
                )
                validation_results[section.section_id] = section_result

                if not section_result["is_valid"]:
                    overall_valid = False

        return {
            "overall_valid": overall_valid,
            "section_validations": validation_results,
            "template_id": template_id,
            "template_name": template.name,
        }

    def _validate_section_content(
        self, section: ReportSection, content: Any
    ) -> Dict[str, Any]:
        """Validate section content"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        # Check word limit
        if section.word_limit:
            content_str = str(content)
            word_count = len(content_str.split())
            if word_count > section.word_limit:
                validation_result["warnings"].append(
                    f"Word count exceeded: {word_count}/{section.word_limit}"
                )

        # Check format requirements
        style_req = section.style_requirements

        if style_req.get("requires_tables", False):
            if "|" not in str(content) and "table" not in str(content).lower():
                validation_result["errors"].append("Section requires tables")
                validation_result["is_valid"] = False

        if style_req.get("requires_charts", False):
            chart_keywords = ["chart", "chart", "graph", "plot"]
            if not any(keyword in str(content).lower() for keyword in chart_keywords):
                validation_result["suggestions"].append(
                    "Recommend adding charts to enhance visualization"
                )

        if style_req.get("format") == "bullet_points":
            if not ("•" in str(content) or "-" in str(content) or "*" in str(content)):
                validation_result["suggestions"].append(
                    "Recommend using bullet point format"
                )

        return validation_result

    def generate_report_template(self, template_id: str) -> str:
        """Generate report template Markdown"""
        template = self.get_template(template_id)
        if not template:
            return f"# Error: Template not found {template_id}"

        markdown = f"# {template.name}\n\n"
        markdown += f"*{template.description}*\n\n"

        # Add generation info
        markdown += (
            f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        markdown += f"**Template Type**: {template.domain.value}\n\n"

        markdown += "---\n\n"

        # Generate section templates
        for section in template.sections:
            required_mark = " *(Required)*" if section.required else " *(Optional)*"
            markdown += f"## {section.title}{required_mark}\n\n"

            # Add section description
            if section.word_limit:
                markdown += (
                    f"*Recommended word count: within {section.word_limit} words*\n\n"
                )

            # Add format requirements
            if section.style_requirements:
                req_list = []
                if section.style_requirements.get("requires_tables"):
                    req_list.append("Include tables")
                if section.style_requirements.get("requires_charts"):
                    req_list.append("Include charts")
                if section.style_requirements.get("format") == "bullet_points":
                    req_list.append("Use bullet point format")

                if req_list:
                    markdown += f"*Format requirements: {', '.join(req_list)}*\n\n"

            # Add content placeholder
            markdown += f"<!-- {section.section_id} section content -->\n\n"

            # Add example structure based on section type
            if section.section_type == SectionType.EXECUTIVE_SUMMARY:
                markdown += "### Key Points\n\n"
                markdown += "- Point one\n- Point two\n- Point three\n\n"
            elif section.section_type == SectionType.ANALYSIS:
                if section.style_requirements.get("requires_tables"):
                    markdown += "### Data Analysis\n\n"
                    markdown += "| Metric | Value | Change |\n"
                    markdown += "|------|------|------|\n"
                    markdown += "| Sample Metric | Sample Value | Sample Change |\n\n"
            elif section.section_type == SectionType.REFERENCES:
                markdown += "### References\n\n"
                markdown += "- [Reference 1](URL1)\n\n"
                markdown += "- [Reference 2](URL2)\n\n"

            markdown += "---\n\n"

        return markdown

    def apply_domain_specific_enhancements(
        self, content: str, domain: ReportDomain
    ) -> str:
        """Apply domain-specific enhancements"""
        enhanced_content = content

        if domain == ReportDomain.FINANCIAL:
            enhanced_content = self._enhance_financial_content(enhanced_content)
        elif domain == ReportDomain.ACADEMIC:
            enhanced_content = self._enhance_academic_content(enhanced_content)
        elif domain == ReportDomain.MARKET_RESEARCH:
            enhanced_content = self._enhance_market_content(enhanced_content)

        return enhanced_content

    def _enhance_financial_content(self, content: str) -> str:
        """Enhance financial content"""
        # Add financial terminology standardization
        financial_replacements = {
            "income": "operating income",
            "profit": "net profit",
            "growth": "year-over-year growth",
            "decline": "year-over-year decline",
        }

        enhanced = content
        for old, new in financial_replacements.items():
            enhanced = enhanced.replace(old, new)

        return enhanced

    def _enhance_academic_content(self, content: str) -> str:
        """Enhance academic content"""
        # Add academic writing standards
        if "I think" in content:
            content = content.replace("I think", "Research shows")
        if "We found" in content:
            content = content.replace("We found", "Research found")

        return content

    def _enhance_market_content(self, content: str) -> str:
        """Enhance market research content"""
        # Add market terminology standardization
        market_replacements = {
            "users": "target user groups",
            "competitors": "main competitors",
            "trends": "market trends",
        }

        enhanced = content
        for old, new in market_replacements.items():
            enhanced = enhanced.replace(old, new)

        return enhanced


# Usage Example
def demo_template_engine():
    """Demonstrate template engine"""
    builder = ReportBuilder()

    # Get financial template
    template = builder.get_template("financial_analysis")
    if template:
        print(f"Template: {template.name}")
        print(f"Domain: {template.domain.value}")
        print(f"Section count: {len(template.sections)}")

        # Generate template
        template_markdown = builder.generate_report_template("financial_analysis")
        print("\nGenerated template:")
        print(template_markdown[:500] + "...")

        # Validate sample content
        sample_content = {
            "exec_summary": "This report analyzes the company's financial condition...",
            "financial_overview": (
                "| Metric | 2023 | 2024 |\n|------|------|------|\n| Revenue | 100M | 115M |"
            ),
            "performance_analysis": (
                "The company's performance is excellent, with 15% year-over-year growth"
            ),
            "risk_assessment": (
                "Main risks include market volatility and policy changes"
            ),
            "recommendations": "Recommend increasing holdings of this stock",
        }

        validation = builder.validate_report_content(
            "financial_analysis", sample_content
        )
        print(f"\nValidation result: {validation['overall_valid']}")

        # Apply domain enhancement
        enhanced = builder.apply_domain_specific_enhancements(
            sample_content["performance_analysis"], ReportDomain.FINANCIAL
        )
        print(f"\nEnhanced content: {enhanced}")


if __name__ == "__main__":
    demo_template_engine()
