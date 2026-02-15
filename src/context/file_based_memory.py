"""File-based external memory system for context engineering.

This module implements the "filesystem as context" principle from Manus AI,
providing persistent storage for research state that survives context resets.

The three-file pattern:
- task_plan.md: Track phases and progress
- findings.md: Store research and findings
- progress.md: Session log and test results
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


@dataclass
class TaskPlan:
    """Task plan file - task_plan.md

    Tracks research phases and progress with checkboxes.
    Implements the "attention manipulation" principle by keeping
    goals visible and up-to-date.
    """

    research_topic: str
    phases: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Convert to Markdown format for persistence."""
        lines = [
            f"# 研究计划: {self.research_topic}",
            "",
            f"**创建时间**: {self.created_at.isoformat()}",
            f"**更新时间**: {self.updated_at.isoformat()}",
            "",
            "## 研究阶段",
            "",
        ]

        for i, phase in enumerate(self.phases, 1):
            status = "✅" if phase.get("completed") else "⬜"
            lines.append(f"{status} **阶段 {i}**: {phase.get('title', '未命名')}")
            if phase.get("description"):
                lines.append(f"   - {phase['description']}")
            if phase.get("steps"):
                for step in phase["steps"]:
                    step_status = "✅" if step.get("completed") else "⬜"
                    lines.append(f"   {step_status} {step.get('title', '未命名步骤')}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, content: str) -> "TaskPlan":
        """Parse from Markdown content."""
        lines = content.split("\n")
        research_topic = ""
        phases = []
        created_at = datetime.now()
        updated_at = datetime.now()

        for i, line in enumerate(lines):
            if line.startswith("# 研究计划:"):
                research_topic = line.replace("# 研究计划:", "").strip()
            elif line.startswith("**创建时间**:"):
                try:
                    created_at = datetime.fromisoformat(
                        line.split("**创建时间**:")[1].strip()
                    )
                except (ValueError, IndexError):
                    pass
            elif line.startswith("**更新时间**:"):
                try:
                    updated_at = datetime.fromisoformat(
                        line.split("**更新时间**:")[1].strip()
                    )
                except (ValueError, IndexError):
                    pass
            elif line.startswith("⬜ **阶段") or line.startswith("✅ **阶段"):
                match = re.match(r"(⬜|✅) \*\*阶段 (\d+)\*\*: (.+)", line)
                if match:
                    status, _, title = match.groups()
                    phases.append(
                        {"title": title, "completed": status == "✅", "steps": []}
                    )
            elif line.strip().startswith("⬜ ") or line.strip().startswith("✅ "):
                step_match = re.match(r"\s*(⬜|✅) (.+)", line)
                if step_match and phases:
                    status, step_title = step_match.groups()
                    phases[-1]["steps"].append(
                        {"title": step_title, "completed": status == "✅"}
                    )

        return cls(
            research_topic=research_topic,
            phases=phases,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class Findings:
    """Research findings file - findings.md

    Stores observations and key insights from research.
    Implements the "preserve error content" principle by keeping
    all findings for model learning.
    """

    observations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)

    def add_observation(
        self,
        content: str,
        source: str,
        relevance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an observation with source tracking."""
        observation = {
            "content": content,
            "source": source,
            "relevance": relevance,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            observation["metadata"] = metadata
        self.observations.append(observation)

    def add_key_insight(self, insight: str) -> None:
        """Add a key insight."""
        if insight not in self.key_insights:
            self.key_insights.append(insight)

    def add_source(self, url: str, title: str, source_type: str = "web") -> None:
        """Add a source reference."""
        self.sources.append(
            {
                "url": url,
                "title": title,
                "type": source_type,
                "added_at": datetime.now().isoformat(),
            }
        )

    def to_markdown(self) -> str:
        """Convert to Markdown format for persistence."""
        lines = ["# 研究发现", ""]

        if self.key_insights:
            lines.append("## 关键洞察")
            for insight in self.key_insights:
                lines.append(f"- {insight}")
            lines.append("")

        if self.sources:
            lines.append("## 来源列表")
            for source in self.sources:
                lines.append(
                    f"- [{source.get('title', '未知')}]({source.get('url', '')})"
                )
            lines.append("")

        lines.append("## 观察结果")
        for obs in self.observations:
            lines.append(f"### {obs.get('source', '未知来源')}")
            lines.append(f"{obs.get('content', '')}")
            lines.append(f"*相关性: {obs.get('relevance', 1.0):.2f}*")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, content: str) -> "Findings":
        """Parse from Markdown content."""
        findings = cls()
        lines = content.split("\n")
        current_section = None
        current_observation = None

        for line in lines:
            if line.startswith("## 关键洞察"):
                current_section = "insights"
            elif line.startswith("## 来源列表"):
                current_section = "sources"
            elif line.startswith("## 观察结果"):
                current_section = "observations"
            elif line.startswith("### "):
                if current_section == "observations":
                    if current_observation:
                        findings.observations.append(current_observation)
                    current_observation = {"source": line[4:].strip()}
            elif current_section == "insights" and line.startswith("- "):
                findings.key_insights.append(line[2:].strip())
            elif current_section == "sources" and line.startswith("- ["):
                match = re.match(r"- \[(.+)\]\((.+)\)", line)
                if match:
                    findings.sources.append(
                        {"title": match.group(1), "url": match.group(2)}
                    )
            elif current_section == "observations" and current_observation:
                if line.startswith("*相关性:"):
                    match = re.search(r"([\d.]+)", line)
                    if match:
                        current_observation["relevance"] = float(match.group(1))
                elif line.strip():
                    current_observation["content"] = (
                        current_observation.get("content", "") + line + "\n"
                    )

        if current_observation:
            findings.observations.append(current_observation)

        return findings


@dataclass
class Progress:
    """Progress log file - progress.md

    Session log with error tracking for learning.
    Implements the "preserve error content" principle.
    """

    entries: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, str]] = field(default_factory=list)

    def log_action(
        self,
        action: str,
        result: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an action with result."""
        entry = {
            "action": action,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata
        self.entries.append(entry)

    def log_error(
        self,
        error: str,
        context: str,
        recovery: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        """Log an error with context (preserve error content principle)."""
        error_record = {
            "error": error,
            "context": context,
            "recovery": recovery,
            "error_type": error_type or "unknown",
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_record)

    def log_decision(self, decision: str, rationale: str) -> None:
        """Log a key decision with rationale."""
        self.decisions.append(
            {
                "decision": decision,
                "rationale": rationale,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_recent_actions(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent actions."""
        return self.entries[-count:] if len(self.entries) > count else self.entries

    def get_recent_errors(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent errors for context injection."""
        return self.errors[-count:] if len(self.errors) > count else self.errors

    def to_markdown(self) -> str:
        """Convert to Markdown format for persistence."""
        lines = ["# 研究进度", ""]

        lines.append("## 执行日志")
        for entry in self.get_recent_actions(20):
            status = "✅" if entry.get("success") else "❌"
            lines.append(
                f"- {status} [{entry.get('timestamp', '')}] {entry.get('action', '')}"
            )
            if not entry.get("success"):
                lines.append(f"  - 结果: {entry.get('result', '')}")
        lines.append("")

        if self.errors:
            lines.append("## 错误记录（保留用于学习）")
            for err in self.errors:
                lines.append(
                    f"- ❌ [{err.get('error_type', 'unknown')}] {err.get('error', '')}"
                )
                lines.append(f"  - 上下文: {err.get('context', '')[:200]}")
                if err.get("recovery"):
                    lines.append(f"  - 恢复策略: {err['recovery']}")
            lines.append("")

        if self.decisions:
            lines.append("## 关键决策")
            for dec in self.decisions:
                lines.append(f"- {dec.get('decision', '')}")
                lines.append(f"  - 理由: {dec.get('rationale', '')}")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, content: str) -> "Progress":
        """Parse from Markdown content."""
        progress = cls()
        lines = content.split("\n")
        current_section = None

        for line in lines:
            if line.startswith("## 执行日志"):
                current_section = "actions"
            elif line.startswith("## 错误记录"):
                current_section = "errors"
            elif line.startswith("## 关键决策"):
                current_section = "decisions"
            elif current_section == "actions" and line.startswith("- "):
                match = re.match(r"- (✅|❌) \[([^\]]+)\] (.+)", line)
                if match:
                    progress.entries.append(
                        {
                            "success": match.group(1) == "✅",
                            "timestamp": match.group(2),
                            "action": match.group(3),
                        }
                    )
            elif current_section == "errors" and line.startswith("- ❌"):
                match = re.match(r"- ❌ \[([^\]]+)\] (.+)", line)
                if match:
                    progress.errors.append(
                        {"error_type": match.group(1), "error": match.group(2)}
                    )
            elif current_section == "decisions" and line.startswith("- "):
                progress.decisions.append({"decision": line[2:].strip()})

        return progress


class FileBasedMemory:
    """File-based external memory system.

    Implements the "filesystem as context" principle from Manus AI.
    The filesystem serves as unlimited external memory, surviving
    context resets and enabling long-running research tasks.

    Key benefits:
    - Size unlimited (unlike context window)
    - Naturally persistent
    - Agent can directly manipulate
    - Enables session recovery
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.task_plan: Optional[TaskPlan] = None
        self.findings: Findings = Findings()
        self.progress: Progress = Progress()

    @property
    def task_plan_path(self) -> Path:
        """Path to task plan file."""
        return self.workspace / "task_plan.md"

    @property
    def findings_path(self) -> Path:
        """Path to findings file."""
        return self.workspace / "findings.md"

    @property
    def progress_path(self) -> Path:
        """Path to progress file."""
        return self.workspace / "progress.md"

    def initialize(self, research_topic: str, initial_plan: Dict[str, Any]) -> None:
        """Initialize research memory with topic and plan."""
        self.task_plan = TaskPlan(
            research_topic=research_topic, phases=initial_plan.get("phases", [])
        )
        self._persist_all()

    def _persist_all(self) -> None:
        """Persist all files to disk."""
        if self.task_plan:
            self.task_plan_path.write_text(
                self.task_plan.to_markdown(), encoding="utf-8"
            )
        self.findings_path.write_text(self.findings.to_markdown(), encoding="utf-8")
        self.progress_path.write_text(self.progress.to_markdown(), encoding="utf-8")

    def persist(self) -> None:
        """Persist current state to disk."""
        self._persist_all()

    def update_phase_status(self, phase_idx: int, completed: bool) -> None:
        """Update phase completion status."""
        if self.task_plan and 0 <= phase_idx < len(self.task_plan.phases):
            self.task_plan.phases[phase_idx]["completed"] = completed
            self.task_plan.updated_at = datetime.now()
            self.task_plan_path.write_text(
                self.task_plan.to_markdown(), encoding="utf-8"
            )

    def update_step_status(
        self, phase_idx: int, step_idx: int, completed: bool
    ) -> None:
        """Update step completion status within a phase."""
        if self.task_plan and 0 <= phase_idx < len(self.task_plan.phases):
            phase = self.task_plan.phases[phase_idx]
            steps = phase.get("steps", [])
            if 0 <= step_idx < len(steps):
                steps[step_idx]["completed"] = completed
                self.task_plan.updated_at = datetime.now()
                self.task_plan_path.write_text(
                    self.task_plan.to_markdown(), encoding="utf-8"
                )

    def get_current_phase(self) -> Optional[Dict[str, Any]]:
        """Get the current (incomplete) phase."""
        if not self.task_plan:
            return None
        for phase in self.task_plan.phases:
            if not phase.get("completed"):
                return phase
        return None

    def get_context_summary(self) -> str:
        """Get context summary for injection into LLM context.

        Implements the "attention manipulation" principle by
        summarizing current state for model attention.
        """
        summary_parts = []

        current_phase = self.get_current_phase()
        if current_phase:
            summary_parts.append(f"**当前阶段**: {current_phase.get('title', '未知')}")

            incomplete_steps = [
                s for s in current_phase.get("steps", []) if not s.get("completed")
            ]
            if incomplete_steps:
                summary_parts.append("**待完成步骤**:")
                for step in incomplete_steps[:3]:
                    summary_parts.append(f"- {step.get('title', '未知步骤')}")

        if self.findings.key_insights:
            summary_parts.append("**关键洞察**:")
            for insight in self.findings.key_insights[-3:]:
                summary_parts.append(f"- {insight}")

        recent_errors = self.progress.get_recent_errors(3)
        if recent_errors:
            summary_parts.append("**需避免的错误**:")
            for err in recent_errors:
                summary_parts.append(f"- {err.get('error', '未知错误')}")

        return "\n".join(summary_parts)

    def get_attention_reminder(self) -> str:
        """Generate attention reminder for goal consistency.

        Implements the "restate to manipulate attention" principle.
        """
        if not self.task_plan:
            return ""

        completed_phases = sum(1 for p in self.task_plan.phases if p.get("completed"))
        total_phases = len(self.task_plan.phases)
        current_phase = self.get_current_phase()

        reminder = f"""## 🎯 目标提醒

**原始目标**: {self.task_plan.research_topic}

**当前焦点**: {current_phase.get('title', '已完成') if current_phase else '所有阶段已完成'}

**进度**: {completed_phases}/{total_phases} 阶段已完成

---
*请确保所有行动都与原始目标保持一致。*
"""
        return reminder

    def load_from_disk(self) -> bool:
        """Load state from disk files.

        Returns True if successfully loaded, False otherwise.
        """
        try:
            if self.task_plan_path.exists():
                content = self.task_plan_path.read_text(encoding="utf-8")
                self.task_plan = TaskPlan.from_markdown(content)

            if self.findings_path.exists():
                content = self.findings_path.read_text(encoding="utf-8")
                self.findings = Findings.from_markdown(content)

            if self.progress_path.exists():
                content = self.progress_path.read_text(encoding="utf-8")
                self.progress = Progress.from_markdown(content)

            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all memory and delete files."""
        self.task_plan = None
        self.findings = Findings()
        self.progress = Progress()

        for path in [self.task_plan_path, self.findings_path, self.progress_path]:
            if path.exists():
                path.unlink()

    def export_state(self) -> Dict[str, Any]:
        """Export current state as dictionary."""
        return {
            "task_plan": {
                "research_topic": (
                    self.task_plan.research_topic if self.task_plan else ""
                ),
                "phases": self.task_plan.phases if self.task_plan else [],
                "created_at": (
                    self.task_plan.created_at.isoformat() if self.task_plan else None
                ),
                "updated_at": (
                    self.task_plan.updated_at.isoformat() if self.task_plan else None
                ),
            },
            "findings": {
                "observations": self.findings.observations,
                "sources": self.findings.sources,
                "key_insights": self.findings.key_insights,
            },
            "progress": {
                "entries": self.progress.entries,
                "errors": self.progress.errors,
                "decisions": self.progress.decisions,
            },
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from dictionary."""
        if "task_plan" in state:
            tp = state["task_plan"]
            self.task_plan = TaskPlan(
                research_topic=tp.get("research_topic", ""),
                phases=tp.get("phases", []),
                created_at=(
                    datetime.fromisoformat(tp["created_at"])
                    if tp.get("created_at")
                    else datetime.now()
                ),
                updated_at=(
                    datetime.fromisoformat(tp["updated_at"])
                    if tp.get("updated_at")
                    else datetime.now()
                ),
            )

        if "findings" in state:
            f = state["findings"]
            self.findings = Findings(
                observations=f.get("observations", []),
                sources=f.get("sources", []),
                key_insights=f.get("key_insights", []),
            )

        if "progress" in state:
            p = state["progress"]
            self.progress = Progress(
                entries=p.get("entries", []),
                errors=p.get("errors", []),
                decisions=p.get("decisions", []),
            )

        self._persist_all()
