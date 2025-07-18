"""
Prompt Manager for standardized prompt management.
Handles loading, caching, and providing prompts to agents.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import glob

logger = logging.getLogger(__name__)


class PromptManager:
    """Centralized prompt management system."""

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize the prompt manager.

        Args:
            prompts_dir: Directory containing prompt files. Defaults to src/prompts/.
        """
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            self.prompts_dir = Path(__file__).parent

        self._prompts: Dict[str, str] = {}
        self._loaded = False

    def load_prompts(self) -> None:
        """Load all prompt files from the prompts directory."""
        self._prompts.clear()

        # Load main prompts
        self._load_prompts_from_directory(self.prompts_dir)

        # Load subdirectory prompts
        subdirs = ["podcast", "ppt", "prompt_enhancer", "prose"]
        for subdir in subdirs:
            subdir_path = self.prompts_dir / subdir
            if subdir_path.exists():
                self._load_prompts_from_directory(subdir_path, prefix=f"{subdir}_")

        self._loaded = True
        logger.info(f"Loaded {len(self._prompts)} prompts from {self.prompts_dir}")

    def _load_prompts_from_directory(self, directory: Path, prefix: str = "") -> None:
        """Load prompts from a specific directory.

        Args:
            directory: Directory to load prompts from.
            prefix: Prefix to add to prompt names for namespacing.
        """
        if not directory.exists():
            logger.warning(f"Prompts directory does not exist: {directory}")
            return

        # Find all .md files
        pattern = str(directory / "*.md")
        for file_path in glob.glob(pattern):
            try:
                file_path = Path(file_path)
                prompt_name = prefix + file_path.stem

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    self._prompts[prompt_name] = content
                    logger.debug(f"Loaded prompt: {prompt_name}")
                else:
                    logger.warning(f"Empty prompt file: {file_path}")

            except Exception as e:
                logger.error(f"Failed to load prompt file {file_path}: {e}")

    def get_prompt(self, prompt_name: str, default: Optional[str] = None) -> str:
        """Get a prompt by name.

        Args:
            prompt_name: Name of the prompt to retrieve.
            default: Default prompt to return if not found.

        Returns:
            Prompt content.

        Raises:
            KeyError: If prompt not found and no default provided.
        """
        if not self._loaded:
            self.load_prompts()

        if prompt_name in self._prompts:
            return self._prompts[prompt_name]

        if default is not None:
            return default

        raise KeyError(f"Prompt '{prompt_name}' not found")

    def get_prompt_with_variables(
        self, prompt_name: str, variables: Dict[str, Any], default: Optional[str] = None
    ) -> str:
        """Get a prompt with variable substitution.

        Args:
            prompt_name: Name of the prompt to retrieve.
            variables: Dictionary of variables to substitute in the prompt.
            default: Default prompt to return if not found.

        Returns:
            Prompt content with variables substituted.
        """
        prompt = self.get_prompt(prompt_name, default)

        try:
            return prompt.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in prompt {prompt_name}: {e}")
            return prompt

    def list_prompts(self) -> list[str]:
        """List all available prompt names.

        Returns:
            List of prompt names.
        """
        if not self._loaded:
            self.load_prompts()

        return list(self._prompts.keys())

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists.

        Args:
            prompt_name: Name of the prompt to check.

        Returns:
            True if prompt exists, False otherwise.
        """
        if not self._loaded:
            self.load_prompts()

        return prompt_name in self._prompts

    def reload_prompts(self) -> None:
        """Reload all prompts from disk."""
        self._loaded = False
        self.load_prompts()

    def add_prompt(self, prompt_name: str, content: str) -> None:
        """Add or update a prompt dynamically.

        Args:
            prompt_name: Name of the prompt.
            content: Prompt content.
        """
        if not self._loaded:
            self.load_prompts()

        self._prompts[prompt_name] = content
        logger.info(f"Added/updated prompt: {prompt_name}")

    def remove_prompt(self, prompt_name: str) -> bool:
        """Remove a prompt.

        Args:
            prompt_name: Name of the prompt to remove.

        Returns:
            True if prompt was removed, False if it didn't exist.
        """
        if not self._loaded:
            self.load_prompts()

        if prompt_name in self._prompts:
            del self._prompts[prompt_name]
            logger.info(f"Removed prompt: {prompt_name}")
            return True

        return False

    def get_prompt_metadata(self, prompt_name: str) -> Dict[str, Any]:
        """Get metadata for a prompt.

        Args:
            prompt_name: Name of the prompt.

        Returns:
            Dictionary containing prompt metadata.
        """
        if not self._loaded:
            self.load_prompts()

        if prompt_name not in self._prompts:
            return {}

        content = self._prompts[prompt_name]

        return {
            "name": prompt_name,
            "length": len(content),
            "lines": len(content.splitlines()),
            "has_placeholders": "{" in content and "}" in content,
            "preview": content[:200] + "..." if len(content) > 200 else content,
        }


# Global prompt manager instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance.

    Returns:
        Prompt manager instance.
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(prompt_name: str, default: Optional[str] = None) -> str:
    """Convenience function to get a prompt.

    Args:
        prompt_name: Name of the prompt to retrieve.
        default: Default prompt to return if not found.

    Returns:
        Prompt content.
    """
    manager = get_prompt_manager()
    return manager.get_prompt(prompt_name, default)


def get_prompt_with_variables(
    prompt_name: str, variables: Dict[str, Any], default: Optional[str] = None
) -> str:
    """Convenience function to get a prompt with variable substitution.

    Args:
        prompt_name: Name of the prompt to retrieve.
        variables: Dictionary of variables to substitute in the prompt.
        default: Default prompt to return if not found.

    Returns:
        Prompt content with variables substituted.
    """
    manager = get_prompt_manager()
    return manager.get_prompt_with_variables(prompt_name, variables, default)
