"""Diversity injection for avoiding few-shot traps.

This module implements the "avoid few-shot traps" principle from Manus AI,
introducing controlled variation to prevent pattern fixation.

Key principles:
1. Introduce structured variation
2. Use alternative phrasing
3. Add controlled noise
4. Shuffle order while preserving meaning
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DiversityConfig:
    """Configuration for diversity injection."""

    template_variations: int = 3
    noise_level: float = 0.1
    shuffle_order: bool = True
    max_intro_variations: int = 5
    max_conclusion_variations: int = 5


class DiversityInjector:
    """Diversity injector - implements "avoid few-shot traps" principle.

    Language models are excellent imitators - they mimic behavior patterns
    in context. If context is filled with similar past action-observation
    pairs, the model will tend to follow that pattern even when suboptimal.

    Solution: Introduce controlled variation.
    """

    INTRO_VARIATIONS = [
        "以下是相关信息：",
        "相关内容如下：",
        "请参考以下信息：",
        "根据搜索结果：",
        "信息摘要：",
    ]

    CONCLUSION_VARIATIONS = [
        "请基于以上信息进行分析。",
        "根据上述内容进行判断。",
        "以上是可供参考的信息。",
        "请结合以上信息继续研究。",
        "以上信息供参考。",
    ]

    SEARCH_RESULT_TEMPLATES = [
        "**来源 {num}**: {title}\n\n{content}",
        "### {title}\n\n{content}",
        "{num}. **{title}**\n\n{content}",
        "【{title}】\n\n{content}",
    ]

    def __init__(self, config: Optional[DiversityConfig] = None):
        self.config = config or DiversityConfig()
        self._template_cache: Dict[str, List[str]] = {}

    def inject_template_variation(
        self, base_template: str, variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Inject template variation.

        Key principles:
        - Introduce small structured changes
        - Different serialization templates
        - Alternative phrasing
        - Minor noise in order or format

        Args:
            base_template: Base template string
            variables: Variables to substitute

        Returns:
            Varied template string
        """
        variables = variables or {}
        result = base_template

        if "{intro}" in result:
            variations = self.INTRO_VARIATIONS[: self.config.max_intro_variations]
            result = result.replace("{intro}", random.choice(variations))

        if "{conclusion}" in result:
            variations = self.CONCLUSION_VARIATIONS[
                : self.config.max_conclusion_variations
            ]
            result = result.replace("{conclusion}", random.choice(variations))

        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        return result

    def add_controlled_noise(self, content: str, noise_type: str = "whitespace") -> str:
        """Add controlled noise to content.

        Args:
            content: Content to add noise to
            noise_type: Type of noise to add

        Returns:
            Content with noise
        """
        if noise_type == "whitespace":
            lines = content.split("\n")
            result = []
            for line in lines:
                if random.random() < self.config.noise_level:
                    line = line + " " * random.randint(0, 2)
                result.append(line)
            return "\n".join(result)

        elif noise_type == "punctuation":
            if random.random() < self.config.noise_level:
                content = content.replace("。", "。 ")
                content = content.replace("，", "， ")
            return content

        return content

    def shuffle_order_preserving_meaning(
        self, items: List[Any], preserve_first: bool = True, preserve_last: bool = False
    ) -> List[Any]:
        """Shuffle order while preserving semantic meaning.

        Args:
            items: Items to shuffle
            preserve_first: Keep first item in place
            preserve_last: Keep last item in place

        Returns:
            Shuffled list
        """
        if len(items) <= 2:
            return items.copy()

        result = items.copy()

        if preserve_first and preserve_last:
            middle = result[1:-1]
            random.shuffle(middle)
            return [result[0]] + middle + [result[-1]]
        elif preserve_first:
            first = result[0]
            rest = result[1:]
            random.shuffle(rest)
            return [first] + rest
        elif preserve_last:
            last = result[-1]
            rest = result[:-1]
            random.shuffle(rest)
            return rest + [last]
        else:
            random.shuffle(result)
            return result

    def vary_search_results(
        self,
        results: List[Dict[str, Any]],
        include_intro: bool = True,
        include_conclusion: bool = True,
    ) -> str:
        """Create varied search results presentation.

        Args:
            results: Search results list
            include_intro: Include intro variation
            include_conclusion: Include conclusion variation

        Returns:
            Formatted search results string
        """
        if not results:
            return ""

        shuffled = self.shuffle_order_preserving_meaning(results, preserve_first=True)

        parts = []

        if include_intro:
            parts.append(random.choice(self.INTRO_VARIATIONS))
            parts.append("")

        for i, result in enumerate(shuffled, 1):
            template = random.choice(self.SEARCH_RESULT_TEMPLATES)
            formatted = template.format(
                num=i,
                title=result.get("title", "未知"),
                content=result.get("content", "")[:500],
            )
            parts.append(formatted)
            parts.append("")

        if include_conclusion:
            parts.append(random.choice(self.CONCLUSION_VARIATIONS))

        return "\n".join(parts)

    def vary_observation_format(
        self, observations: List[str], max_per_format: int = 5
    ) -> str:
        """Vary observation formatting.

        Args:
            observations: List of observations
            max_per_format: Max observations per format

        Returns:
            Varied observation string
        """
        if not observations:
            return ""

        formats = [
            lambda obs, i: f"{i}. {obs}",
            lambda obs, i: f"- {obs}",
            lambda obs, i: f"• {obs}",
            lambda obs, i: f"【{i}】{obs}",
        ]

        shuffled = self.shuffle_order_preserving_meaning(
            observations, preserve_first=False
        )

        selected_format = random.choice(formats)

        lines = []
        for i, obs in enumerate(shuffled[:max_per_format], 1):
            lines.append(selected_format(obs, i))

        return "\n".join(lines)

    def inject_phasing_variation(self, text: str) -> str:
        """Inject phrasing variation.

        Args:
            text: Text to vary

        Returns:
            Varied text
        """
        replacements = {
            "请": ["请", "请", "请尝试"],  # Weighted towards original
            "搜索": ["搜索", "查找", "检索"],
            "分析": ["分析", "研究", "考察"],
            "结果": ["结果", "输出", "返回"],
        }

        result = text
        for original, variations in replacements.items():
            if original in result and random.random() < 0.3:
                result = result.replace(original, random.choice(variations), 1)

        return result

    def create_diverse_context(
        self,
        search_results: List[Dict[str, Any]],
        observations: List[str],
        include_diversity: bool = True,
    ) -> str:
        """Create diverse context from multiple sources.

        Args:
            search_results: Search results
            observations: Observations list
            include_diversity: Whether to apply diversity

        Returns:
            Formatted context string
        """
        if not include_diversity:
            parts = []
            for i, r in enumerate(search_results, 1):
                parts.append(
                    f"**来源 {i}**: {r.get('title', '')}\n\n{r.get('content', '')}"
                )
            for obs in observations:
                parts.append(f"- {obs}")
            return "\n\n---\n\n".join(parts)

        parts = []

        if search_results:
            parts.append(self.vary_search_results(search_results))

        if observations:
            parts.append("## 观察结果")
            parts.append(self.vary_observation_format(observations))

        return "\n\n".join(parts)


def build_context_with_diversity(
    observations: List[str],
    search_results: List[Dict],
    config: Optional[DiversityConfig] = None,
) -> str:
    """Build context with diversity injection.

    Convenience function for building diverse context.

    Args:
        observations: List of observations
        search_results: List of search results
        config: Diversity configuration

    Returns:
        Diverse context string
    """
    injector = DiversityInjector(config)
    return injector.create_diverse_context(search_results, observations)


def get_random_intro() -> str:
    """Get a random intro phrase."""
    return random.choice(DiversityInjector.INTRO_VARIATIONS)


def get_random_conclusion() -> str:
    """Get a random conclusion phrase."""
    return random.choice(DiversityInjector.CONCLUSION_VARIATIONS)
