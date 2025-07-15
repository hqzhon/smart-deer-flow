#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置系统迁移清理脚本

此脚本用于:
1. 删除旧的配置文件和代码
2. 更新所有配置引用到新的统一配置系统
3. 生成迁移报告
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import re


class ConfigMigrationCleaner:
    """配置迁移清理器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.migration_report = []
        
        # 需要删除的旧配置文件
        self.files_to_delete = [
            "src/utils/config/config_management.py",
            "src/utils/config/cleaning_config.py", 
            "src/config/researcher_config_loader.py",
            "src/config/parallel_config.py",
            "src/config/configuration.py",  # 旧的配置文件，已被新系统替代
            "src/config/config_manager.py",  # 旧的配置管理器
            "src/config/config_integration.py",  # 旧的配置集成
            "src/config/loader.py",  # 旧的加载器
            "src/utils/config",  # 整个旧配置目录
        ]
        
        # 需要更新的文件和对应的替换规则
        self.files_to_update = {
            "main.py": [
                (r"from src\.config import load_configuration, get_settings", 
                 "from src.config import get_settings"),
                (r"load_configuration\(.*?\)", 
                 "# Configuration loaded automatically via DI"),
            ],
            "src/llms/llm.py": [
                (r"from src\.config import load_yaml_config",
                 "from src.config import get_settings"),
                (r"load_yaml_config\(.*?\)",
                 "get_settings().llm"),
            ],
            "src/utils/system/rate_limiter.py": [
                (r"from src\.config\.config_loader import config_loader",
                 "from src.config import get_settings"),
                (r"config_loader\.(.*?)",
                 "get_settings().\\1"),
            ],
            "src/utils/performance/parallel_executor.py": [
                (r"from src\.config\.config_loader import config_loader",
                 "from src.config import get_settings"),
                (r"config_loader\.(.*?)",
                 "get_settings().\\1"),
            ],
            "src/utils/template.py": [
                (r"from src\.config\.config_loader import config_loader",
                 "from src.config import get_settings"),
                (r"config_loader\.(.*?)",
                 "get_settings().\\1"),
            ],
            "src/utils/context/context_evaluator.py": [
                (r"from src\.config\.config_loader import config_loader",
                 "from src.config import get_settings"),
                (r"config_loader\.(.*?)",
                 "get_settings().\\1"),
            ],
            "src/graph/nodes.py": [
                (r"from src\.config\.config_loader import config_loader",
                 "from src.config import get_settings"),
                (r"config_loader\.(.*?)",
                 "get_settings().\\1"),
            ],
            "src/config/__init__.py": [
                (r"from \.new_config_loader import load_configuration, get_settings, get_config_loader",
                 "from .config_loader import get_settings"),
                (r"\"load_configuration\",\s*", ""),
                (r"\"get_config_loader\",\s*", ""),
            ],
        }
        
        # 需要更新配置引用的文件
        self.config_reference_updates = {
            "src/config/config_service.py": [
                (r"from \.config_loader import config_loader",
                 "from .config_loader import get_settings"),
                (r"config_loader\.", "get_settings()."),
                (r"from \.config_manager import ConfigManager", ""),
                (r"ConfigManager\(.*?\)", "get_settings()"),
            ],
            "src/utils/__init__.py": [
                (r"from \.config import ConfigManager, CleaningConfig", ""),
                (r"ConfigManager", "# Removed - use get_settings() instead"),
                (r"CleaningConfig", "# Removed - use get_settings() instead"),
            ],
        }
    
    def run_migration(self) -> Dict[str, List[str]]:
        """运行完整的迁移清理"""
        report = {
            "deleted_files": [],
            "updated_files": [],
            "errors": [],
            "warnings": []
        }
        
        print("开始配置系统迁移清理...")
        
        # 1. 删除旧配置文件
        report["deleted_files"] = self._delete_old_files()
        
        # 2. 更新文件引用
        report["updated_files"] = self._update_file_references()
        
        # 3. 检查剩余的旧配置引用
        report["warnings"] = self._check_remaining_references()
        
        # 4. 生成迁移报告
        self._generate_migration_report(report)
        
        print("配置系统迁移清理完成!")
        return report
    
    def _delete_old_files(self) -> List[str]:
        """删除旧的配置文件"""
        deleted_files = []
        
        for file_path in self.files_to_delete:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    if full_path.is_file():
                        full_path.unlink()
                    else:
                        shutil.rmtree(full_path)
                    deleted_files.append(str(file_path))
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
            else:
                print(f"文件不存在: {file_path}")
        
        return deleted_files
    
    def _update_file_references(self) -> List[str]:
        """更新文件中的配置引用"""
        updated_files = []
        
        # 更新主要文件
        for file_path, replacements in self.files_to_update.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for pattern, replacement in replacements:
                        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    
                    if content != original_content:
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        updated_files.append(str(file_path))
                        print(f"已更新: {file_path}")
                        
                except Exception as e:
                    print(f"更新失败 {file_path}: {e}")
        
        # 更新配置引用
        for file_path, replacements in self.config_reference_updates.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for pattern, replacement in replacements:
                        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    
                    if content != original_content:
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        updated_files.append(str(file_path))
                        print(f"已更新配置引用: {file_path}")
                        
                except Exception as e:
                    print(f"更新配置引用失败 {file_path}: {e}")
        
        return updated_files
    
    def _check_remaining_references(self) -> List[str]:
        """检查剩余的旧配置引用"""
        warnings = []
        
        # 检查可能的旧配置引用模式
        old_patterns = [
            r"load_yaml_config",
            r"config_loader\.",
            r"load_configuration",
            r"ConfigManager",
            r"CleaningConfig",
            r"ResearcherConfigLoader",
        ]
        
        # 扫描源代码目录
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in old_patterns:
                        if re.search(pattern, content):
                            rel_path = py_file.relative_to(self.project_root)
                            warnings.append(f"在 {rel_path} 中发现可能的旧配置引用: {pattern}")
                            
                except Exception as e:
                    continue
        
        return warnings
    
    def _generate_migration_report(self, report: Dict[str, List[str]]) -> None:
        """生成迁移报告"""
        report_content = [
            "# 配置系统迁移清理报告",
            f"生成时间: {os.popen('date').read().strip()}",
            "",
            "## 已删除的文件",
        ]
        
        for file in report["deleted_files"]:
            report_content.append(f"- {file}")
        
        report_content.extend([
            "",
            "## 已更新的文件",
        ])
        
        for file in report["updated_files"]:
            report_content.append(f"- {file}")
        
        if report["warnings"]:
            report_content.extend([
                "",
                "## 警告 - 需要手动检查",
            ])
            
            for warning in report["warnings"]:
                report_content.append(f"- {warning}")
        
        report_content.extend([
            "",
            "## 迁移后的配置系统",
            "",
            "新的统一配置系统位于:",
            "- `src/config/` - 核心配置模块",
            "- `src/config/models.py` - Pydantic配置模型",
            "- `src/config/config_loader.py` - 配置加载器",
            "- `src/config/config_service.py` - 配置服务",
            "- `src/config/di_config.py` - 依赖注入配置",
            "",
            "使用方式:",
            "```python",
            "from src.config import get_settings",
            "settings = get_settings()",
            "```",
        ])
        
        report_file = self.project_root / "MIGRATION_CLEANUP_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        
        print(f"迁移报告已生成: {report_file}")


if __name__ == "__main__":
    cleaner = ConfigMigrationCleaner()
    report = cleaner.run_migration()
    
    print("\n=== 迁移摘要 ===")
    print(f"删除文件: {len(report['deleted_files'])}")
    print(f"更新文件: {len(report['updated_files'])}")
    print(f"警告: {len(report['warnings'])}")
    
    if report["warnings"]:
        print("\n请检查警告中的文件，可能需要手动更新配置引用。")