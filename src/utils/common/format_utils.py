# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


def format_bytes(size: int) -> str:
    """Format bytes to human readable string.

    Args:
        size: Size in bytes

    Returns:
        Human readable string like "1.5GB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"
