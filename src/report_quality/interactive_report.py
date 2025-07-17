# DeerFlow Report Quality Optimization - Interactive Report Output Feature

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import hashlib
import os

from .i18n import Language, get_text


@dataclass
class InteractiveReportConfig:
    """Configuration for interactive report generation"""

    output_dir: Optional[str] = None
    filename_template: str = "interactive_report_{language}_{timestamp}.html"
    auto_create_dirs: bool = True
    encoding: str = "utf-8"
    use_timestamp: bool = True

    def get_output_path(self, language: Language, report_id: str = None) -> str:
        """Get the full output path for a given language"""
        from datetime import datetime

        # Prepare template variables
        template_vars = {
            "language": language.value,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "report_id": report_id or f"report_{uuid.uuid4().hex[:8]}",
        }

        # If timestamp is disabled, use simple template
        if not self.use_timestamp:
            filename = f"interactive_report_{language.value}.html"
        else:
            filename = self.filename_template.format(**template_vars)

        if self.output_dir:
            if self.auto_create_dirs:
                os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, filename)
        else:
            return filename


logger = logging.getLogger(__name__)


class InteractiveElementType(Enum):
    """Interactive element types"""

    CLICKABLE_CHART = "clickable_chart"
    EXPANDABLE_SECTION = "expandable_section"
    DATA_DRILL_DOWN = "data_drill_down"
    SOURCE_LINK = "source_link"
    CODE_VIEWER = "code_viewer"
    DYNAMIC_TABLE = "dynamic_table"
    FILTER_CONTROL = "filter_control"
    TOOLTIP = "tooltip"


@dataclass
class InteractiveElement:
    """Interactive element"""

    element_id: str
    element_type: InteractiveElementType
    title: str
    description: str
    target_content: str
    trigger_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSource:
    """Data source information"""

    source_id: str
    name: str
    description: str
    file_path: Optional[str] = None
    query_code: Optional[str] = None
    raw_data: Optional[Any] = None
    last_updated: Optional[str] = None


@dataclass
class CodeBlock:
    """Code block"""

    code_id: str
    language: str
    code: str
    description: str
    output: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class InteractiveReport:
    """Interactive report"""

    report_id: str
    title: str
    content: str
    interactive_elements: List[InteractiveElement] = field(default_factory=list)
    data_sources: List[DataSource] = field(default_factory=list)
    code_blocks: List[CodeBlock] = field(default_factory=list)
    navigation_tree: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InteractiveElementGenerator:
    """Interactive element generator"""

    def __init__(self, language: Language = Language.ZH_CN):
        self.language = language
        self.element_patterns = self._get_element_patterns()

    def _get_element_patterns(self) -> Dict[InteractiveElementType, List[str]]:
        """Get element patterns based on language"""
        if self.language == Language.ZH_CN:
            return {
                InteractiveElementType.CLICKABLE_CHART: [
                    r"!\[.*?\]\(.*?\.(?:png|jpg|jpeg|svg)\)",  # Standard markdown images
                    r"!\s+https?://[^\s]+",  # New simple format: ! https://...
                    r"图\s*\d+",
                    r"图表\s*\d+",
                    r"chart|graph|plot",
                ],
                InteractiveElementType.DATA_DRILL_DOWN: [
                    r"\|.*?\|.*?\|",  # Tables
                    r"数据显示",
                    r"统计结果",
                    r"分析数据",
                ],
                InteractiveElementType.SOURCE_LINK: [
                    r"\[.*?\]\(https?://.*?\)",  # Links
                    r"来源[:：]",
                    r"数据来源",
                    r"参考文献",
                ],
                InteractiveElementType.CODE_VIEWER: [
                    r"```.*?```",  # Code blocks
                    r"SQL查询",
                    r"Python代码",
                    r"分析代码",
                ],
            }
        else:  # English
            return {
                InteractiveElementType.CLICKABLE_CHART: [
                    r"!\[.*?\]\(.*?\.(?:png|jpg|jpeg|svg)\)",  # Standard markdown images
                    r"!\s+https?://[^\s]+",  # New simple format: ! https://...
                    r"figure\s*\d+",
                    r"chart\s*\d+",
                    r"chart|graph|plot",
                ],
                InteractiveElementType.DATA_DRILL_DOWN: [
                    r"\|.*?\|.*?\|",  # Tables
                    r"data shows",
                    r"statistics",
                    r"analysis data",
                ],
                InteractiveElementType.SOURCE_LINK: [
                    r"\[.*?\]\(https?://.*?\)",  # Links
                    r"source[::]",
                    r"data source",
                    r"references",
                ],
                InteractiveElementType.CODE_VIEWER: [
                    r"```.*?```",  # Code blocks
                    r"SQL query",
                    r"Python code",
                    r"analysis code",
                ],
            }

    def extract_interactive_elements(self, content: str) -> List[InteractiveElement]:
        """Extract interactive elements"""
        elements = []
        content_for_links = content

        # Extract chart elements first
        chart_elements = self._extract_chart_elements(content)
        elements.extend(chart_elements)

        # Remove chart trigger texts from content for link extraction
        for chart_element in chart_elements:
            content_for_links = content_for_links.replace(
                chart_element.trigger_text, ""
            )

        # Extract table elements
        table_elements = self._extract_table_elements(content)
        elements.extend(table_elements)

        # Remove table trigger texts from content for link extraction
        for table_element in table_elements:
            content_for_links = content_for_links.replace(
                table_element.trigger_text, ""
            )

        # Extract code elements
        code_elements = self._extract_code_elements(content)
        elements.extend(code_elements)

        # Remove code trigger texts from content for link extraction
        for code_element in code_elements:
            content_for_links = content_for_links.replace(code_element.trigger_text, "")

        # Extract link elements from remaining content
        link_elements = self._extract_link_elements(content_for_links)
        elements.extend(link_elements)

        # Remove duplicate elements based on trigger_text
        unique_elements = []
        seen_trigger_texts = set()

        for element in elements:
            if element.trigger_text not in seen_trigger_texts:
                unique_elements.append(element)
                seen_trigger_texts.add(element.trigger_text)

        return unique_elements

    def _extract_chart_elements(self, content: str) -> List[InteractiveElement]:
        """Extract chart elements"""
        elements = []

        # Find standard markdown image references: ![alt](url)
        img_pattern = r"!\[(.*?)\]\((.*?)\)"
        matches = re.finditer(img_pattern, content)

        for match in matches:
            alt_text = match.group(1)
            img_path = match.group(2)

            # Only process chart-type images
            chart_keywords = (
                ["chart", "graph", "plot", "图表", "图"]
                if self.language == Language.ZH_CN
                else ["chart", "graph", "plot", "figure"]
            )
            if any(keyword in alt_text.lower() for keyword in chart_keywords):
                element = InteractiveElement(
                    element_id=self._generate_element_id("chart"),
                    element_type=InteractiveElementType.CLICKABLE_CHART,
                    title=get_text(
                        self.language, "interactive_elements", "clickable_chart_title"
                    ).format(alt_text=alt_text),
                    description=get_text(
                        self.language,
                        "interactive_elements",
                        "clickable_chart_description",
                    ),
                    target_content=img_path,
                    trigger_text=match.group(0),
                    metadata={
                        "alt_text": alt_text,
                        "image_path": img_path,
                        "chart_type": self._detect_chart_type(alt_text),
                    },
                )
                elements.append(element)

        # Find new format image references: ! https://...
        simple_img_pattern = r"!\s+(https?://[^\s]+)"
        simple_matches = re.finditer(simple_img_pattern, content)

        for match in simple_matches:
            img_path = match.group(1)

            # Extract filename or use URL as alt text
            alt_text = self._extract_filename_from_url(img_path) or "Image"

            # Create element for simple format images (assume they could be charts)
            element = InteractiveElement(
                element_id=self._generate_element_id("chart"),
                element_type=InteractiveElementType.CLICKABLE_CHART,
                title=get_text(
                    self.language, "interactive_elements", "clickable_chart_title"
                ).format(alt_text=alt_text),
                description=get_text(
                    self.language, "interactive_elements", "clickable_chart_description"
                ),
                target_content=img_path,
                trigger_text=match.group(0),
                metadata={
                    "alt_text": alt_text,
                    "image_path": img_path,
                    "chart_type": self._detect_chart_type(alt_text),
                    "format_type": "simple",
                },
            )
            elements.append(element)

        return elements

    def _extract_table_elements(self, content: str) -> List[InteractiveElement]:
        """Extract table elements"""
        elements = []

        # Remove code blocks to avoid false positives
        content_without_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

        # Find Markdown tables
        table_pattern = r"(\|.*?\|.*?\n(?:\|.*?\|.*?\n)*)"
        matches = re.finditer(table_pattern, content_without_code, re.MULTILINE)

        for i, match in enumerate(matches):
            table_content = match.group(1)

            # Additional validation: ensure it's a real table (has at least 2 rows)
            lines = table_content.strip().split("\n")
            if len(lines) < 2:
                continue

            element = InteractiveElement(
                element_id=self._generate_element_id("table"),
                element_type=InteractiveElementType.DYNAMIC_TABLE,
                title=get_text(
                    self.language, "interactive_elements", "dynamic_table_title"
                ).format(index=i + 1),
                description=get_text(
                    self.language, "interactive_elements", "dynamic_table_description"
                ),
                target_content=table_content,
                trigger_text=table_content,
                metadata={
                    "table_index": i,
                    "row_count": len(table_content.split("\n"))
                    - 2,  # Subtract header and separator rows
                    "supports_sorting": True,
                    "supports_filtering": True,
                },
            )
            elements.append(element)

        return elements

    def _extract_link_elements(self, content: str) -> List[InteractiveElement]:
        """Extract link elements (excluding image links and links inside tables)"""
        elements = []

        # First, identify table regions to exclude links within them
        table_regions = []
        table_pattern = r"(\|.*?\|.*?\n(?:\|.*?\|.*?\n)*)"
        table_matches = re.finditer(table_pattern, content, re.MULTILINE)
        for table_match in table_matches:
            table_regions.append((table_match.start(), table_match.end()))

        # Find Markdown links, but exclude image links (![alt](url))
        link_pattern = r"(?<!!)\[(.*?)\]\((https?://.*?)\)"
        matches = re.finditer(link_pattern, content)

        for match in matches:
            link_text = match.group(1)
            url = match.group(2)

            # Additional check: skip if this looks like an image URL
            if any(
                url.lower().endswith(ext)
                for ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]
            ):
                continue

            # Check if this link is inside a table
            link_start = match.start()
            link_end = match.end()
            is_in_table = any(
                table_start <= link_start < table_end
                and table_start < link_end <= table_end
                for table_start, table_end in table_regions
            )

            # Skip links that are inside tables
            if is_in_table:
                continue

            element = InteractiveElement(
                element_id=self._generate_element_id("link"),
                element_type=InteractiveElementType.SOURCE_LINK,
                title=get_text(
                    self.language, "interactive_elements", "source_link_title"
                ).format(link_text=link_text),
                description=get_text(
                    self.language, "interactive_elements", "source_link_description"
                ),
                target_content=url,
                trigger_text=match.group(0),
                metadata={
                    "link_text": link_text,
                    "url": url,
                    "domain": self._extract_domain(url),
                    "is_external": True,
                },
            )
            elements.append(element)

        return elements

    def _extract_code_elements(self, content: str) -> List[InteractiveElement]:
        """Extract code elements"""
        elements = []

        # Find code blocks
        code_pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.finditer(code_pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            language = match.group(1) or "text"
            code_content = match.group(2)

            element = InteractiveElement(
                element_id=self._generate_element_id("code"),
                element_type=InteractiveElementType.CODE_VIEWER,
                title=get_text(
                    self.language, "interactive_elements", "code_viewer_title"
                ).format(language=language.upper()),
                description=get_text(
                    self.language, "interactive_elements", "code_viewer_description"
                ),
                target_content=code_content,
                trigger_text=match.group(0),
                metadata={
                    "language": language,
                    "code_length": len(code_content),
                    "is_executable": language in ["python", "sql", "r"],
                    "code_index": i,
                },
            )
            elements.append(element)

        return elements

    def _generate_element_id(self, prefix: str) -> str:
        """Generate element ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _detect_chart_type(self, alt_text: str) -> str:
        """Detect chart type"""
        alt_lower = alt_text.lower()

        if any(keyword in alt_lower for keyword in ["bar", "柱状", "条形"]):
            return "bar_chart"
        elif any(keyword in alt_lower for keyword in ["line", "折线", "趋势"]):
            return "line_chart"
        elif any(keyword in alt_lower for keyword in ["pie", "饼图", "圆饼"]):
            return "pie_chart"
        elif any(keyword in alt_lower for keyword in ["scatter", "散点", "散布"]):
            return "scatter_plot"
        else:
            return "unknown"

    def _extract_domain(self, url: str) -> str:
        """Extract domain"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc
        except Exception:
            return "unknown"

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        try:
            from urllib.parse import urlparse

            path = urlparse(url).path
            filename = path.split("/")[-1]
            # Remove file extension for cleaner alt text
            if "." in filename:
                filename = filename.rsplit(".", 1)[0]
            return filename.replace("-", " ").replace("_", " ").title()
        except Exception:
            return "Image"


class ReportEnhancer:
    """Report enhancer"""

    def __init__(
        self,
        language: Language = Language.ZH_CN,
        config: InteractiveReportConfig = None,
    ):
        self.language = language
        self.config = config or InteractiveReportConfig()
        self.element_generator = InteractiveElementGenerator(language)

    def enhance_report(
        self, content: str, metadata: Dict[str, Any] = None
    ) -> InteractiveReport:
        """Enhance report to interactive version"""
        if metadata is None:
            metadata = {}

        report_id = self._generate_report_id()

        # Extract interactive elements
        interactive_elements = self.element_generator.extract_interactive_elements(
            content
        )

        # Generate data source information
        data_sources = self._extract_data_sources(content, metadata)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)

        # Generate navigation tree
        navigation_tree = self._generate_navigation_tree(content)

        # Enhance content
        enhanced_content = self._enhance_content_with_interactivity(
            content, interactive_elements
        )

        return InteractiveReport(
            report_id=report_id,
            title=metadata.get(
                "title",
                get_text(self.language, "interactive_elements", "default_report_title"),
            ),
            content=enhanced_content,
            interactive_elements=interactive_elements,
            data_sources=data_sources,
            code_blocks=code_blocks,
            navigation_tree=navigation_tree,
            metadata=metadata,
        )

    def generate_html_report(
        self, content: str, metadata: Dict[str, Any] = None, output_dir: str = None
    ) -> str:
        """Generate and save interactive HTML report

        Args:
            content: Report content in markdown format
            metadata: Report metadata
            output_dir: Output directory for generated files. If None, uses config or current directory.

        Returns:
            Path to the generated HTML file
        """
        # Generate interactive report
        interactive_report = self.enhance_report(content, metadata)

        # Generate HTML
        html_generator = HTMLGenerator(language=self.language)
        html_content = html_generator.generate_interactive_html(interactive_report)

        # Determine output path
        if output_dir:
            # Use provided output_dir, override config
            temp_config = InteractiveReportConfig(
                output_dir=output_dir, use_timestamp=self.config.use_timestamp
            )
            output_file = temp_config.get_output_path(
                self.language, interactive_report.report_id
            )
        else:
            # Use instance config
            output_file = self.config.get_output_path(
                self.language, interactive_report.report_id
            )

        # Save HTML file
        with open(output_file, "w", encoding=self.config.encoding) as f:
            f.write(html_content)

        return output_file

    def _generate_report_id(self) -> str:
        """Generate report ID"""
        return f"report_{uuid.uuid4().hex[:12]}"

    def _extract_data_sources(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[DataSource]:
        """Extract data sources"""
        sources = []
        seen_urls = set()  # Track seen URLs to avoid duplicates

        # Extract from metadata
        if "data_sources" in metadata:
            for i, source_name in enumerate(metadata["data_sources"]):
                source = DataSource(
                    source_id=f"ds_{i:03d}",
                    name=source_name,
                    description=get_text(
                        self.language, "interactive_elements", "data_source_description"
                    ).format(name=source_name),
                    last_updated=metadata.get("last_updated"),
                )
                sources.append(source)

        # Extract links from content as data sources
        link_pattern = r"\[(.*?)\]\((https?://.*?)\)"
        matches = re.finditer(link_pattern, content)

        for match in matches:
            link_text = match.group(1)
            url = match.group(2)

            # Skip if we've already seen this URL
            if url in seen_urls:
                continue
            seen_urls.add(url)

            source = DataSource(
                source_id=f"ds_{len(sources):03d}",
                name=link_text,
                description=get_text(
                    self.language,
                    "interactive_elements",
                    "external_data_source_description",
                ).format(name=link_text),
                file_path=url,
                last_updated=None,
            )
            sources.append(source)

        return sources

    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract code blocks"""
        blocks = []

        code_pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.finditer(code_pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            language = match.group(1) or "text"
            code = match.group(2).strip()

            block = CodeBlock(
                code_id=f"code_{i:03d}",
                language=language,
                code=code,
                description=get_text(
                    self.language, "interactive_elements", "code_block_description"
                ).format(language=language.upper(), index=i + 1),
                dependencies=self._extract_dependencies(code, language),
            )
            blocks.append(block)

        return blocks

    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract code dependencies"""
        dependencies = []

        if language.lower() == "python":
            # Extract import statements
            import_pattern = r"(?:import|from)\s+(\w+)"
            matches = re.findall(import_pattern, code)
            dependencies.extend(matches)

        elif language.lower() == "sql":
            # Extract table names
            table_pattern = r"FROM\s+(\w+)|JOIN\s+(\w+)"
            matches = re.findall(table_pattern, code, re.IGNORECASE)
            for match in matches:
                dependencies.extend([t for t in match if t])

        return list(set(dependencies))  # Remove duplicates

    def _generate_navigation_tree(self, content: str) -> Dict[str, Any]:
        """Generate navigation tree"""
        tree = {"sections": []}

        # Extract heading levels (支持1-6级标题)
        heading_pattern = r"^(#{1,6})\s+(.*?)$"
        matches = re.finditer(heading_pattern, content, re.MULTILINE)

        current_section = None
        current_subsection = None

        for match in matches:
            level = len(match.group(1))
            title = match.group(2).strip()

            section = {
                "level": level,
                "title": title,
                "id": self._generate_section_id(title),
                "subsections": [],
            }

            if level == 1:
                tree["sections"].append(section)
                current_section = section
                current_subsection = None
            elif level == 2 and current_section:
                current_section["subsections"].append(section)
                current_subsection = section
            elif level == 3 and current_subsection:
                current_subsection["subsections"].append(section)
            elif (
                level == 4 and current_subsection and current_subsection["subsections"]
            ):
                # 四级标题作为三级标题的子项
                if current_subsection["subsections"]:
                    last_l3 = current_subsection["subsections"][-1]
                    if "subsections" not in last_l3:
                        last_l3["subsections"] = []
                    last_l3["subsections"].append(section)

        return tree

    def _generate_section_id(self, title: str) -> str:
        """Generate section ID"""
        # Convert to URL-friendly ID
        section_id = re.sub(r"[^\w\u4e00-\u9fff]+", "-", title.lower())
        section_id = section_id.strip("-")
        return section_id or "section"

    def _enhance_content_with_interactivity(
        self, content: str, elements: List[InteractiveElement]
    ) -> str:
        """Enhance content with interactivity"""
        enhanced = content

        # Sort elements by trigger_text length (longest first) to avoid partial matches
        sorted_elements = sorted(
            elements, key=lambda x: len(x.trigger_text), reverse=True
        )

        # Track replaced trigger texts to avoid duplicates
        replaced_triggers = set()

        # Apply replacements one by one, checking for overlaps in the current content
        for element in sorted_elements:
            # Skip if this trigger text has already been replaced
            if element.trigger_text in replaced_triggers:
                continue

            # Find the trigger text in current enhanced content
            pos = enhanced.find(element.trigger_text)
            if pos != -1:
                # Check if this position is already inside an interactive tag
                before_pos = enhanced.rfind("<interactive", 0, pos)
                after_pos = enhanced.find("</interactive>", pos)

                # If we found an opening tag before and closing tag after, skip this replacement
                if before_pos != -1 and after_pos != -1:
                    # Check if there's a closing tag between the opening tag and our position
                    closing_between = enhanced.find("</interactive>", before_pos, pos)
                    if closing_between == -1:
                        # We're inside an interactive tag, skip this replacement
                        continue

                # Apply the replacement
                interactive_marker = f'<interactive id="{element.element_id}" type="{element.element_type.value}" title="{element.title}">'
                replacement_text = (
                    f"{interactive_marker}{element.trigger_text}</interactive>"
                )
                enhanced = (
                    enhanced[:pos]
                    + replacement_text
                    + enhanced[pos + len(element.trigger_text) :]
                )

                # Mark this trigger as replaced
                replaced_triggers.add(element.trigger_text)

        return enhanced


class HTMLGenerator:
    """HTML generator"""

    def __init__(self, language: Language = Language.ZH_CN):
        self.language = language
        self.template_dir = Path(__file__).parent / "templates"

    def generate_interactive_html(self, report: InteractiveReport) -> str:
        """Generate interactive HTML"""
        html_template = self._get_html_template()

        # Replace template variables
        lang_code = "zh-CN" if self.language == Language.ZH_CN else "en-US"
        html = html_template.replace("{lang_code}", lang_code)
        html = html.replace("{{TITLE}}", report.title)
        html = html.replace(
            "{{CONTENT}}", self._convert_markdown_to_html(report.content)
        )
        html = html.replace(
            "{{NAVIGATION}}", self._generate_navigation_html(report.navigation_tree)
        )
        html = html.replace(
            "{{INTERACTIVE_ELEMENTS}}",
            self._generate_elements_js(report.interactive_elements),
        )
        html = html.replace(
            "{{DATA_SOURCES}}", self._generate_data_sources_html(report.data_sources)
        )
        html = html.replace(
            "{{CODE_BLOCKS}}", self._generate_code_blocks_html(report.code_blocks)
        )
        html = html.replace(
            "{{DATA_SOURCES_TITLE}}",
            get_text(self.language, "interactive_elements", "data_sources"),
        )
        html = html.replace(
            "{{CODE_BLOCKS_TITLE}}",
            get_text(self.language, "interactive_elements", "code_blocks"),
        )

        return html

    def _get_html_template(self) -> str:
        """Get HTML template"""
        lang_code = "zh-CN" if self.language == Language.ZH_CN else "en-US"
        return """
<!DOCTYPE html>
<html lang="{lang_code}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .navigation {
            background: #34495e;
            color: white;
            padding: 10px;
        }
        .navigation ul {
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .navigation li {
            display: inline-block;
            margin-right: 20px;
        }
        .navigation a {
            color: white;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 3px;
        }
        .navigation a:hover {
            background: #3498db;
        }
        .content {
            padding: 30px;
        }
        .interactive {
            position: relative;
            cursor: pointer;
            border: 2px dashed #3498db;
            margin: 10px 0;
            padding: 10px;
            background: #ecf0f1;
            transition: all 0.3s ease;
        }
        .interactive:hover {
            background: #d5dbdb;
            border-color: #2980b9;
        }
        .interactive::after {
            content: "🔗 Click for details";
            position: absolute;
            top: -10px;
            right: 10px;
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: white;
            margin: 15% auto;
            padding: 20px;
            border-radius: 5px;
            width: 80%;
            max-width: 800px;
            position: relative;
        }
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover {
            color: black;
        }
        .sidebar {
            position: fixed;
            right: -300px;
            top: 0;
            width: 300px;
            height: 100%;
            background: white;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
            transition: right 0.3s ease;
            z-index: 999;
            overflow-y: auto;
        }
        .sidebar.open {
            right: 0;
        }
        .sidebar-toggle {
            position: fixed;
            right: 20px;
            top: 20px;
            background: #3498db;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 50%;
            cursor: pointer;
            z-index: 1000;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            cursor: pointer;
        }
        th:hover {
            background-color: #e6e6e6;
        }
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            overflow-x: auto;
        }
        .code-block pre {
            margin: 0;
            font-family: 'Courier New', monospace;
        }
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #3498db, #2c3e50, #3498db);
            margin: 30px 0;
            border-radius: 1px;
        }
        .mermaid-diagram {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            overflow-x: auto;
        }
        .mermaid {
            font-family: 'trebuchet ms', verdana, arial, sans-serif;
            font-size: 16px;
            fill: #333;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{TITLE}}</h1>
        </div>
        
        <div class="navigation">
            {{NAVIGATION}}
        </div>
        
        <div class="content">
            {{CONTENT}}
        </div>
    </div>
    
    <button class="sidebar-toggle" onclick="toggleSidebar()">📊</button>
    
    <div class="sidebar" id="sidebar">
        <h3>{{DATA_SOURCES_TITLE}}</h3>
        {{DATA_SOURCES}}
        
        <h3>{{CODE_BLOCKS_TITLE}}</h3>
        {{CODE_BLOCKS}}
    </div>
    
    <div class="modal" id="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modal-body"></div>
        </div>
    </div>
    
    <script>
        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
        
        {{INTERACTIVE_ELEMENTS}}
        
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.toggle('open');
        }
        
        function openModal(content) {
            const modal = document.getElementById('modal');
            const modalBody = document.getElementById('modal-body');
            modalBody.innerHTML = content;
            modal.style.display = 'block';
        }
        
        function closeModal() {
            const modal = document.getElementById('modal');
            modal.style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('modal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        
        // Table sorting functionality
        function sortTable(table, column) {
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            const isNumeric = !isNaN(parseFloat(rows[0].cells[column].textContent));
            
            rows.sort((a, b) => {
                const aVal = a.cells[column].textContent;
                const bVal = b.cells[column].textContent;
                
                if (isNumeric) {
                    return parseFloat(aVal) - parseFloat(bVal);
                }
                return aVal.localeCompare(bVal);
            });
            
            const tbody = table.querySelector('tbody');
            rows.forEach(row => tbody.appendChild(row));
        }
    </script>
</body>
</html>
"""

    def _convert_markdown_to_html(self, markdown: str) -> str:
        """Convert Markdown to HTML (simplified version)"""
        html = markdown

        # Horizontal rules (分割线)
        html = re.sub(r"^---+\s*$", r"<hr>", html, flags=re.MULTILINE)

        # Headers (支持1-6级标题)
        html = re.sub(r"^###### (.*?)$", r"<h6>\1</h6>", html, flags=re.MULTILINE)
        html = re.sub(r"^##### (.*?)$", r"<h5>\1</h5>", html, flags=re.MULTILINE)
        html = re.sub(r"^#### (.*?)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Lists (支持-号列表)
        html = self._convert_lists_to_html(html)

        # Images (must be processed before regular links)
        html = re.sub(r"!\[(.*?)\]\((.*?)\)", r'<img src="\2" alt="\1" />', html)

        # Links (regular links, not images)
        html = re.sub(
            r"\[(.*?)\]\((.*?)\)", r'<a href="\2" target="_blank">\1</a>', html
        )

        # Bold
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)

        # Paragraphs
        html = re.sub(r"\n\n", "</p><p>", html)
        html = f"<p>{html}</p>"

        # Tables
        html = self._convert_tables_to_html(html)

        # Code blocks and Mermaid diagrams
        html = self._convert_code_blocks_to_html(html)

        return html

    def _convert_lists_to_html(self, content: str) -> str:
        """Convert Markdown lists to HTML"""
        lines = content.split("\n")
        result = []
        in_list = False
        list_level = 0

        for line in lines:
            # 检测列表项 (支持 - 和 * 开头的列表)
            list_match = re.match(r"^(\s*)[-*]\s+(.*)", line)

            if list_match:
                indent = len(list_match.group(1))
                content = list_match.group(2)
                current_level = indent // 2  # 假设每级缩进2个空格

                if not in_list:
                    result.append("<ul>")
                    in_list = True
                    list_level = current_level
                elif current_level > list_level:
                    # 嵌套列表
                    result.append("<ul>")
                    list_level = current_level
                elif current_level < list_level:
                    # 结束嵌套
                    for _ in range(list_level - current_level):
                        result.append("</ul>")
                    list_level = current_level

                result.append(f"<li>{content}</li>")
            else:
                if in_list:
                    # 结束列表
                    for _ in range(list_level + 1):
                        result.append("</ul>")
                    in_list = False
                    list_level = 0
                result.append(line)

        # 如果文件结束时还在列表中，关闭列表
        if in_list:
            for _ in range(list_level + 1):
                result.append("</ul>")

        return "\n".join(result)

    def _convert_tables_to_html(self, content: str) -> str:
        """Convert Markdown tables to HTML"""
        # Simplified table conversion
        table_pattern = r"(\|.*?\|.*?\n(?:\|.*?\|.*?\n)*)"

        def convert_table(match):
            table_md = match.group(1)
            lines = table_md.strip().split("\n")

            if len(lines) < 2:
                return match.group(0)

            # Header row
            header_line = lines[0]
            headers = [cell.strip() for cell in header_line.split("|")[1:-1]]

            # Data rows
            data_lines = lines[2:]  # Skip separator row

            html = '<table class="sortable">\n<thead>\n<tr>\n'
            for i, header in enumerate(headers):
                html += f"<th onclick=\"sortTable(this.closest('table'), {i})\">{header}</th>\n"
            html += "</tr>\n</thead>\n<tbody>\n"

            for line in data_lines:
                if "|" in line:
                    cells = [cell.strip() for cell in line.split("|")[1:-1]]
                    html += "<tr>\n"
                    for cell in cells:
                        html += f"<td>{cell}</td>\n"
                    html += "</tr>\n"

            html += "</tbody>\n</table>\n"
            return html

        return re.sub(table_pattern, convert_table, content, flags=re.MULTILINE)

    def _convert_code_blocks_to_html(self, content: str) -> str:
        """Convert code blocks and Mermaid diagrams to HTML"""

        def convert_code_block(match):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()

            # Check if it's a Mermaid diagram
            if language.lower() == "mermaid" or code_content.strip().startswith(
                (
                    "graph TD",
                    "graph LR",
                    "graph TB",
                    "graph RL",
                    "flowchart TD",
                    "flowchart LR",
                    "sequenceDiagram",
                    "classDiagram",
                    "stateDiagram",
                    "erDiagram",
                    "journey",
                    "gantt",
                    "pie",
                )
            ):
                # Generate unique ID for the diagram
                diagram_id = (
                    f"mermaid-{hashlib.md5(code_content.encode()).hexdigest()[:8]}"
                )
                return f"""
<div class="mermaid-diagram" id="{diagram_id}">
    <div class="mermaid">{code_content}</div>
</div>
"""
            else:
                # Regular code block
                return f'<div class="code-block"><pre><code class="language-{language}">{code_content}</code></pre></div>'

        # Convert code blocks
        html = re.sub(
            r"```(\w+)?\n(.*?)\n```", convert_code_block, content, flags=re.DOTALL
        )
        return html

    def _generate_navigation_html(self, nav_tree: Dict[str, Any]) -> str:
        """Generate navigation HTML"""
        if not nav_tree.get("sections"):
            return f"<ul><li>{get_text(self.language, 'interactive_elements', 'no_navigation_items')}</li></ul>"

        html = "<ul>"
        for section in nav_tree["sections"]:
            html += f'<li><a href="#{section["id"]}">{section["title"]}</a></li>'
        html += "</ul>"

        return html

    def _generate_elements_js(self, elements: List[InteractiveElement]) -> str:
        """Generate interactive elements JavaScript"""
        js = "// Interactive element handling\n"

        for element in elements:
            if element.element_type == InteractiveElementType.CLICKABLE_CHART:
                chart_type = element.metadata.get("chart_type", "Unknown")
                js += f"""
document.addEventListener('DOMContentLoaded', function() {{
    const element = document.querySelector('[id="{element.element_id}"]');
    if (element) {{
        element.addEventListener('click', function() {{
            openModal('<h3>{element.title}</h3><p>{element.description}</p><p><strong>Chart Path:</strong> {element.target_content}</p><p><strong>Type:</strong> {chart_type}</p>');
        }});
    }}
}});
"""
            elif element.element_type == InteractiveElementType.CODE_VIEWER:
                safe_code = element.target_content.replace("'", "\\'")
                js += f"""
document.addEventListener('DOMContentLoaded', function() {{
    const element = document.querySelector('[id="{element.element_id}"]');
    if (element) {{
        element.addEventListener('click', function() {{
            openModal('<h3>{element.title}</h3><p>{element.description}</p><div class="code-block"><pre><code>{safe_code}</code></pre></div>');
        }});
    }}
}});
"""

        return js

    def _generate_data_sources_html(self, sources: List[DataSource]) -> str:
        """Generate data sources HTML"""
        if not sources:
            return f"<p>{get_text(self.language, 'interactive_elements', 'no_data_sources')}</p>"

        html = "<ul>"
        for source in sources:
            html += f"""
            <li>
                <strong>{source.name}</strong><br>
                <small>{source.description}</small>
                {f'<br><a href="{source.file_path}" target="_blank">{get_text(self.language, "interactive_elements", "view_source_file")}</a>' if source.file_path else ''}
            </li>
            """
        html += "</ul>"

        return html

    def _generate_code_blocks_html(self, blocks: List[CodeBlock]) -> str:
        """Generate code blocks HTML"""
        if not blocks:
            return f"<p>{get_text(self.language, 'interactive_elements', 'no_code_blocks')}</p>"

        html = "<ul>"
        for block in blocks:
            html += f'''
            <li>
                <strong>{block.description}</strong><br>
                <small>{get_text(self.language, 'interactive_elements', 'language_label')}: {block.language}</small>
                <button onclick="openModal('<h3>{block.description}</h3><div class=\'code-block\'><pre><code>{block.code.replace("'", "\\'")}></code></pre></div>')">{get_text(self.language, 'interactive_elements', 'view_code_button')}</button>
            </li>
            '''
        html += "</ul>"

        return html


def demo_interactive_report(
    language: Language = Language.ZH_CN, output_dir: str = None
):
    """Demo interactive report generation

    Args:
        language: Report language
        output_dir: Output directory for generated files. If None, uses current directory.
    """

    enhancer = ReportEnhancer(language=language)
    html_generator = HTMLGenerator(language=language)

    # Sample report content
    sample_content = """
# 2023年度财务分析报告

## 执行摘要

本报告分析了公司2023年的财务表现，包括收入增长、成本控制和盈利能力等关键指标。

---

### 主要发现

- 总收入同比增长15%
- 净利润率提升至12.5%
- 运营成本有效控制
  - 人力成本控制在预算内
  - 营销费用优化20%
  - 办公成本降低8%

#### 关键指标详情

具体的关键绩效指标如下：

- ROI提升至18%
- 客户满意度达到95%
- 员工留存率保持在92%

---

## 详细分析

### 收入分析

![收入趋势图](revenue_chart.png "2023年月度收入趋势")

#### 季度收入对比

各季度收入表现如下：

| 季度 | 收入(万元) | 增长率 |
|------|------------|--------|
| Q1   | 1200       | 10%    |
| Q2   | 1350       | 15%    |
| Q3   | 1480       | 18%    |
| Q4   | 1620       | 20%    |

#### 收入构成分析

收入主要来源包括：

- 产品销售收入
  - 核心产品A：占比45%
  - 新产品B：占比30%
  - 其他产品：占比25%
- 服务收入
  - 技术支持：占比60%
  - 咨询服务：占比40%

---

### 技术实现

```python
import pandas as pd
import matplotlib.pyplot as plt

# 数据处理
df = pd.read_csv('financial_data.csv')
revenue_growth = df['revenue'].pct_change()

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(df['quarter'], df['revenue'])
plt.title('季度收入趋势')
plt.show()
```

---

### 数据源

详细数据来源请参考：[财务系统报表](https://finance.company.com/reports)

---

## 结论

基于以上分析，公司在2023年表现优异，建议继续保持当前增长策略。

### 建议措施

- 继续投资核心产品研发
- 扩大市场营销投入
- 优化运营流程
  - 自动化关键业务流程
  - 提升数据分析能力
  - 加强团队协作

#### 短期目标

- Q1目标：收入增长12%
- Q2目标：新产品上线
- Q3目标：市场份额提升5%

#### 长期规划

- 2024年：进入新市场
- 2025年：实现IPO准备
- 2026年：国际化扩张
"""

    # Generate interactive report
    interactive_report = enhancer.enhance_report(
        sample_content,
        metadata={
            "title": "2023 Financial Analysis Report",
            "data_sources": ["财务系统", "销售数据库", "市场调研报告"],
            "last_updated": "2024-01-15",
        },
    )

    # Generate HTML
    html_content = html_generator.generate_interactive_html(interactive_report)

    # Prepare output path using config
    config = InteractiveReportConfig(output_dir=output_dir)
    output_file = config.get_output_path(language, interactive_report.report_id)

    # Save HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Interactive report generated: {output_file}")
    return output_file


if __name__ == "__main__":
    demo_interactive_report()
