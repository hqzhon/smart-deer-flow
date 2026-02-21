from pathlib import Path

from .parser import parse_skill_file
from .types import Skill


def get_skills_root_path() -> Path:
    """
    Get the root path of the skills directory.

    Returns:
        Path to the skills directory (deer-flow/skills)
    """
    backend_dir = Path(__file__).resolve().parent.parent
    skills_dir = backend_dir.parent / "skills"
    return skills_dir


def load_skills(
    skills_path: Path | None = None, use_config: bool = True, enabled_only: bool = False
) -> list[Skill]:
    """
    Load all skills from the skills directory.

    Scans both public and custom skill directories, parsing SKILL.md files
    to extract metadata. The enabled state is determined by the skills_state_config.json file.

    Args:
        skills_path: Optional custom path to skills directory.
                     If not provided and use_config is True, uses path from config.
                     Otherwise defaults to deer-flow/skills
        use_config: Whether to load skills path from config (default: True)
        enabled_only: If True, only return enabled skills (default: False)

    Returns:
        List of Skill objects, sorted by name
    """
    if skills_path is None:
        if use_config:
            try:
                from src.config.config_loader import get_settings

                settings = get_settings()
                if hasattr(settings, "skills") and settings.skills:
                    skills_path = Path(getattr(settings.skills, "skills_path", None))
            except Exception:
                skills_path = get_skills_root_path()
        else:
            skills_path = get_skills_root_path()

    if not skills_path or not skills_path.exists():
        return []

    skills = []

    for category in ["public", "custom"]:
        category_path = skills_path / category
        if not category_path.exists() or not category_path.is_dir():
            continue

        for skill_dir in category_path.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            skill = parse_skill_file(skill_file, category=category)
            if skill:
                skills.append(skill)

    try:
        from src.config.extensions_config import ExtensionsConfig

        extensions_config = ExtensionsConfig.from_file()
        for skill in skills:
            skill.enabled = extensions_config.is_skill_enabled(
                skill.name, skill.category
            )
    except Exception as e:
        print(f"Warning: Failed to load extensions config: {e}")

    if enabled_only:
        skills = [skill for skill in skills if skill.enabled]

    skills.sort(key=lambda s: s.name)

    return skills
