# 报告文件唯一命名功能使用指南

## 概述

为了解决报告生成时文件名固定导致的覆盖问题，我们优化了报告文件命名机制，支持生成唯一的文件名，避免文件覆盖。

## 主要特性

### 1. 时间戳命名
- **默认行为**: 文件名包含时间戳，格式为 `interactive_report_{language}_{timestamp}.html`
- **时间戳格式**: `YYYYMMDD_HHMMSS`（如：`20250707_234541`）
- **优点**: 每次生成都有唯一文件名，不会覆盖历史报告

### 2. 报告ID命名
- 每个报告都有唯一的报告ID（基于UUID）
- 可在自定义模板中使用 `{report_id}` 变量

### 3. 灵活配置
- 支持自定义文件名模板
- 可选择启用/禁用时间戳
- 支持自定义输出目录

## 使用方法

### 基本使用（默认配置）

```python
from src.report_quality.interactive_report import ReportEnhancer
from src.report_quality.i18n import Language

# 创建报告增强器（默认启用时间戳）
enhancer = ReportEnhancer(language=Language.ZH_CN)

# 生成报告
file_path = enhancer.generate_html_report(
    content="# 我的报告\n\n这是报告内容...",
    metadata={"title": "测试报告"},
    output_dir="reports"
)

print(f"报告已生成: {file_path}")
# 输出示例: 报告已生成: reports/interactive_report_zh_cn_20250707_234541.html
```

### 禁用时间戳（传统模式）

```python
from src.report_quality.interactive_report import ReportEnhancer, InteractiveReportConfig
from src.report_quality.i18n import Language

# 创建配置（禁用时间戳）
config = InteractiveReportConfig(
    output_dir="reports",
    use_timestamp=False  # 禁用时间戳
)

# 创建报告增强器
enhancer = ReportEnhancer(language=Language.ZH_CN, config=config)

# 生成报告（会覆盖同名文件）
file_path = enhancer.generate_html_report(
    content="# 我的报告\n\n这是报告内容...",
    metadata={"title": "测试报告"}
)

print(f"报告已生成: {file_path}")
# 输出示例: 报告已生成: reports/interactive_report_zh_cn.html
```

### 自定义文件名模板

```python
from src.report_quality.interactive_report import ReportEnhancer, InteractiveReportConfig
from src.report_quality.i18n import Language

# 创建自定义配置
config = InteractiveReportConfig(
    output_dir="reports",
    filename_template="report_{report_id}_{language}_{timestamp}.html",
    use_timestamp=True
)

# 创建报告增强器
enhancer = ReportEnhancer(language=Language.ZH_CN, config=config)

# 生成报告
file_path = enhancer.generate_html_report(
    content="# 我的报告\n\n这是报告内容...",
    metadata={"title": "自定义模板报告"}
)

print(f"报告已生成: {file_path}")
# 输出示例: 报告已生成: reports/report_report_79d7110b_zh_cn_20250707_234542.html
```

### 仅使用报告ID（不含时间戳）

```python
from src.report_quality.interactive_report import ReportEnhancer, InteractiveReportConfig
from src.report_quality.i18n import Language

# 创建配置
config = InteractiveReportConfig(
    output_dir="reports",
    filename_template="report_{report_id}_{language}.html",
    use_timestamp=False  # 禁用时间戳，但仍使用报告ID
)

# 创建报告增强器
enhancer = ReportEnhancer(language=Language.ZH_CN, config=config)

# 生成报告
file_path = enhancer.generate_html_report(
    content="# 我的报告\n\n这是报告内容...",
    metadata={"title": "报告ID模板"}
)

print(f"报告已生成: {file_path}")
# 输出示例: 报告已生成: reports/report_report_a1b2c3d4_zh_cn.html
```

## 配置参数说明

### InteractiveReportConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | `Optional[str]` | `None` | 输出目录路径，None表示当前目录 |
| `filename_template` | `str` | `"interactive_report_{language}_{timestamp}.html"` | 文件名模板 |
| `auto_create_dirs` | `bool` | `True` | 是否自动创建输出目录 |
| `encoding` | `str` | `"utf-8"` | 文件编码 |
| `use_timestamp` | `bool` | `True` | 是否在文件名中包含时间戳 |

### 文件名模板变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `{language}` | 语言代码 | `zh_cn`, `en_us` |
| `{timestamp}` | 时间戳 | `20250707_234541` |
| `{report_id}` | 报告唯一ID | `report_79d7110b` |

## 最佳实践

### 1. 生产环境推荐配置

```python
# 生产环境：启用时间戳，便于版本管理
config = InteractiveReportConfig(
    output_dir="/path/to/reports",
    filename_template="report_{timestamp}_{language}.html",
    use_timestamp=True,
    auto_create_dirs=True
)
```

### 2. 开发环境推荐配置

```python
# 开发环境：可选择禁用时间戳，便于调试
config = InteractiveReportConfig(
    output_dir="./debug_reports",
    use_timestamp=False  # 方便覆盖测试文件
)
```

### 3. 批量生成推荐配置

```python
# 批量生成：使用报告ID确保唯一性
config = InteractiveReportConfig(
    output_dir="./batch_reports",
    filename_template="batch_{report_id}_{language}.html",
    use_timestamp=False  # 依赖报告ID保证唯一性
)
```

## 迁移指南

### 从旧版本迁移

如果您之前使用的是固定文件名的版本，现在默认会生成带时间戳的文件名。如果需要保持旧的行为：

```python
# 保持旧版本行为
config = InteractiveReportConfig(use_timestamp=False)
enhancer = ReportEnhancer(config=config)
```

### 文件管理建议

1. **定期清理**: 由于启用时间戳后会产生多个文件，建议定期清理旧的报告文件
2. **目录组织**: 建议按日期或项目创建子目录来组织报告文件
3. **备份策略**: 重要报告建议备份到其他位置

## 故障排除

### 常见问题

1. **文件仍然被覆盖**
   - 检查 `use_timestamp` 是否设置为 `True`
   - 确认 `filename_template` 中包含 `{timestamp}` 或 `{report_id}`

2. **文件名格式不符合预期**
   - 检查 `filename_template` 配置
   - 确认模板变量拼写正确

3. **输出目录创建失败**
   - 检查目录权限
   - 确认 `auto_create_dirs` 设置为 `True`

### 调试方法

```python
# 调试文件名生成
config = InteractiveReportConfig()
test_path = config.get_output_path(Language.ZH_CN, "test_report_id")
print(f"生成的文件路径: {test_path}")
```

## 更新日志

- **v1.1.0**: 添加唯一文件名支持
  - 新增时间戳命名机制
  - 新增报告ID支持
  - 新增自定义文件名模板
  - 保持向后兼容性