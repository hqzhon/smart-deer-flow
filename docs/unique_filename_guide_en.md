# Unique Filename Feature User Guide for Report Files

## Overview

To solve the file overwriting issue caused by fixed filenames during report generation, we have optimized the report file naming mechanism to support generating unique filenames and avoid file overwriting.

## Main Features

### 1. Timestamp Naming
- **Default Behavior**: Filenames include timestamps in the format `interactive_report_{language}_{timestamp}.html`
- **Timestamp Format**: `YYYYMMDD_HHMMSS` (e.g., `20250707_234541`)
- **Advantages**: Each generation has a unique filename, preventing overwriting of historical reports

### 2. Report ID Naming
- Each report has a unique report ID (based on UUID)
- The `{report_id}` variable can be used in custom templates

### 3. Flexible Configuration
- Support for custom filename templates
- Option to enable/disable timestamps
- Support for custom output directories

## Usage

### Basic Usage (Default Configuration)

```python
from src.report_quality.interactive_report import ReportEnhancer
from src.report_quality.i18n import Language

# Create report enhancer (timestamp enabled by default)
enhancer = ReportEnhancer(language=Language.ZH_CN)

# Generate report
file_path = enhancer.generate_html_report(
    content="# My Report\n\nThis is the report content...",
    metadata={"title": "Test Report"},
    output_dir="reports"
)

print(f"Report generated: {file_path}")
# Example output: Report generated: reports/interactive_report_zh_cn_20250707_234541.html
```

### Disable Timestamp (Legacy Mode)

```python
from src.report_quality.interactive_report import ReportEnhancer, InteractiveReportConfig
from src.report_quality.i18n import Language

# Create configuration (disable timestamp)
config = InteractiveReportConfig(
    output_dir="reports",
    use_timestamp=False  # Disable timestamp
)

# Create report enhancer
enhancer = ReportEnhancer(language=Language.ZH_CN, config=config)

# Generate report (will overwrite files with same name)
file_path = enhancer.generate_html_report(
    content="# My Report\n\nThis is the report content...",
    metadata={"title": "Test Report"}
)

print(f"Report generated: {file_path}")
# Example output: Report generated: reports/interactive_report_zh_cn.html
```

### Custom Filename Template

```python
from src.report_quality.interactive_report import ReportEnhancer, InteractiveReportConfig
from src.report_quality.i18n import Language

# Create custom configuration
config = InteractiveReportConfig(
    output_dir="reports",
    filename_template="report_{report_id}_{language}_{timestamp}.html",
    use_timestamp=True
)

# Create report enhancer
enhancer = ReportEnhancer(language=Language.ZH_CN, config=config)

# Generate report
file_path = enhancer.generate_html_report(
    content="# My Report\n\nThis is the report content...",
    metadata={"title": "Custom Template Report"}
)

print(f"Report generated: {file_path}")
# Example output: Report generated: reports/report_report_79d7110b_zh_cn_20250707_234542.html
```

### Report ID Only (No Timestamp)

```python
from src.report_quality.interactive_report import ReportEnhancer, InteractiveReportConfig
from src.report_quality.i18n import Language

# Create configuration
config = InteractiveReportConfig(
    output_dir="reports",
    filename_template="report_{report_id}_{language}.html",
    use_timestamp=False  # Disable timestamp but still use report ID
)

# Create report enhancer
enhancer = ReportEnhancer(language=Language.ZH_CN, config=config)

# Generate report
file_path = enhancer.generate_html_report(
    content="# My Report\n\nThis is the report content...",
    metadata={"title": "Report ID Template"}
)

print(f"Report generated: {file_path}")
# Example output: Report generated: reports/report_report_a1b2c3d4_zh_cn.html
```

## Configuration Parameters

### InteractiveReportConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `Optional[str]` | `None` | Output directory path, None means current directory |
| `filename_template` | `str` | `"interactive_report_{language}_{timestamp}.html"` | Filename template |
| `auto_create_dirs` | `bool` | `True` | Whether to automatically create output directory |
| `encoding` | `str` | `"utf-8"` | File encoding |
| `use_timestamp` | `bool` | `True` | Whether to include timestamp in filename |

### Filename Template Variables

| Variable | Description | Example |
|----------|-------------|----------|
| `{language}` | Language code | `zh_cn`, `en_us` |
| `{timestamp}` | Timestamp | `20250707_234541` |
| `{report_id}` | Unique report ID | `report_79d7110b` |

## Best Practices

### 1. Production Environment Recommended Configuration

```python
# Production environment: Enable timestamp for version management
config = InteractiveReportConfig(
    output_dir="/path/to/reports",
    filename_template="report_{timestamp}_{language}.html",
    use_timestamp=True,
    auto_create_dirs=True
)
```

### 2. Development Environment Recommended Configuration

```python
# Development environment: Optionally disable timestamp for debugging convenience
config = InteractiveReportConfig(
    output_dir="./debug_reports",
    use_timestamp=False  # Convenient for overwriting test files
)
```

### 3. Batch Generation Recommended Configuration

```python
# Batch generation: Use report ID to ensure uniqueness
config = InteractiveReportConfig(
    output_dir="./batch_reports",
    filename_template="batch_{report_id}_{language}.html",
    use_timestamp=False  # Rely on report ID for uniqueness
)
```

## Migration Guide

### Migrating from Previous Versions

If you previously used the fixed filename version, the default behavior now generates filenames with timestamps. To maintain the old behavior:

```python
# Maintain old version behavior
config = InteractiveReportConfig(use_timestamp=False)
enhancer = ReportEnhancer(config=config)
```

### File Management Recommendations

1. **Regular Cleanup**: Since enabling timestamps will generate multiple files, regular cleanup of old report files is recommended
2. **Directory Organization**: It's recommended to create subdirectories by date or project to organize report files
3. **Backup Strategy**: Important reports should be backed up to other locations

## Troubleshooting

### Common Issues

1. **Files Still Being Overwritten**
   - Check if `use_timestamp` is set to `True`
   - Ensure `filename_template` includes `{timestamp}` or `{report_id}`

2. **Filename Format Not as Expected**
   - Check `filename_template` configuration
   - Confirm template variables are spelled correctly

3. **Output Directory Creation Failed**
   - Check directory permissions
   - Ensure `auto_create_dirs` is set to `True`

### Debugging Methods

```python
# Debug filename generation
config = InteractiveReportConfig()
test_path = config.get_output_path(Language.ZH_CN, "test_report_id")
print(f"Generated file path: {test_path}")
```

## Changelog

- **v1.1.0**: Added unique filename support
  - Added timestamp naming mechanism
  - Added report ID support
  - Added custom filename template
  - Maintained backward compatibility