"""Content Cleaner - Basic text and content cleaning utilities."""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics for content cleaning operations."""
    total_cleanings: int = 0
    characters_removed: int = 0
    whitespace_normalized: int = 0
    html_tags_removed: int = 0
    comments_removed: int = 0


class ContentCleaner:
    """Basic content cleaner for text, HTML, code, and markdown."""
    
    def __init__(self):
        self.cleaning_rules = {
            'whitespace': r'\s+',
            'multiple_newlines': r'\n{3,}',
            'html_tags': r'<[^>]+>',
            'script_tags': r'<script[^>]*>.*?</script>',
            'style_tags': r'<style[^>]*>.*?</style>',
            'comments': r'#.*?$'
        }
        self.cleaning_stats = CleaningStats()
        self.custom_filters = {}
    
    def clean_text(self, text: str, collect_stats: bool = False) -> str:
        """Clean basic text content."""
        if not text:
            return text
            
        original_length = len(text)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove tabs
        cleaned = cleaned.replace('\t', ' ')
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        if collect_stats:
            self.cleaning_stats.total_cleanings += 1
            self.cleaning_stats.characters_removed += original_length - len(cleaned)
            self.cleaning_stats.whitespace_normalized += 1
            
        return cleaned
    
    def clean_html(self, html_content: str) -> str:
        """Clean HTML content by removing scripts, styles, and tags."""
        if not html_content:
            return html_content
            
        # Remove script tags
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove style tags
        cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags but keep content
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Clean up whitespace
        cleaned = self.clean_text(cleaned)
        
        self.cleaning_stats.html_tags_removed += 1
        
        return cleaned
    
    def clean_code(self, code_content: str, preserve_structure: bool = True, remove_comments: bool = False) -> str:
        """Clean code content."""
        if not code_content:
            return code_content
            
        cleaned = code_content
        
        if remove_comments:
            # Remove Python-style comments
            lines = cleaned.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove inline comments but preserve strings
                if '#' in line and not ('"' in line or "'" in line):
                    line = re.sub(r'#.*$', '', line)
                elif line.strip().startswith('#'):
                    continue
                cleaned_lines.append(line)
            cleaned = '\n'.join(cleaned_lines)
            self.cleaning_stats.comments_removed += 1
        
        if not preserve_structure:
            cleaned = self.clean_text(cleaned)
            
        return cleaned
    
    def clean_markdown(self, markdown_content: str) -> str:
        """Clean markdown content while preserving structure."""
        if not markdown_content:
            return markdown_content
            
        # Basic markdown cleaning - just normalize whitespace
        cleaned = self.clean_text(markdown_content)
        
        return cleaned
    
    def add_custom_rule(self, name: str, pattern: str, replacement: str = '') -> None:
        """Add a custom cleaning rule."""
        self.custom_filters[name] = {'pattern': pattern, 'replacement': replacement}
    
    def apply_custom_rules(self, text: str) -> str:
        """Apply custom cleaning rules."""
        cleaned = text
        for rule_name, rule in self.custom_filters.items():
            cleaned = re.sub(rule['pattern'], rule['replacement'], cleaned)
        return cleaned
    
    def clean_batch(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean multiple content items in batch."""
        results = []
        
        for item in content_items:
            content_type = item.get('type', 'text')
            content = item.get('content', '')
            
            if content_type == 'text':
                cleaned = self.clean_text(content, collect_stats=True)
            elif content_type == 'html':
                cleaned = self.clean_html(content)
            elif content_type == 'code':
                cleaned = self.clean_code(content)
            elif content_type == 'markdown':
                cleaned = self.clean_markdown(content)
            else:
                cleaned = self.clean_text(content)
            
            results.append({
                'cleaned_content': cleaned,
                'original_type': content_type,
                'cleaning_stats': {
                    'original_length': len(content),
                    'cleaned_length': len(cleaned),
                    'reduction_ratio': (len(content) - len(cleaned)) / len(content) if content else 0
                }
            })
        
        return results
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return {
            'total_cleanings': self.cleaning_stats.total_cleanings,
            'characters_removed': self.cleaning_stats.characters_removed,
            'whitespace_normalized': self.cleaning_stats.whitespace_normalized,
            'html_tags_removed': self.cleaning_stats.html_tags_removed,
            'comments_removed': self.cleaning_stats.comments_removed
        }