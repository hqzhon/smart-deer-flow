# DeerFlow 配置系统指南

## 概述

DeerFlow 使用基于 Pydantic 的现代配置系统，提供类型安全、环境变量支持和配置验证功能。

## 配置文件说明

### 当前配置文件状态

- **conf.yaml** - 当前默认配置文件（旧格式）
- **conf_new.yaml** - 新的 Pydantic 配置格式（推荐）
- **conf.yaml.example** - 配置示例文件

### 配置格式对比

#### 旧格式 (conf.yaml)
```yaml
# 旧的配置格式
BASIC_MODEL:
  base_url: https://api.deepseek.com/v1
  model: "deepseek-chat"
  api_key: sk-xxx
  token_limits:
    input_limit: 32000
    output_limit: 4096

max_search_results: 3

PARALLEL_EXECUTION:
  enable_parallel_execution: true
  max_context_steps_parallel: 1
```

#### 新格式 (conf_new.yaml)
```yaml
# 新的 Pydantic 配置格式
report_style: "academic"
resources: []

llm:
  temperature: 0.7
  timeout: 30
  max_tokens: null

agents:
  max_plan_iterations: 1
  max_step_num: 5
  max_search_results: 5
  enable_deep_thinking: false
  enable_parallel_execution: true

research:
  enable_researcher_isolation: true
  researcher_isolation_level: "moderate"

content:
  enable_content_summarization: true
  enable_smart_filtering: true
  summary_type: "comprehensive"
```

## 快速开始

### 一键设置（最简单）

运行配置设置脚本，自动处理所有配置相关任务：
```bash
python setup_config.py
```

这个脚本会：
- 检查现有配置文件
- 自动迁移旧配置（如果需要）
- 验证配置文件
- 设置最佳的默认配置

### 手动迁移

如果您想手动控制迁移过程：

1. 运行迁移脚本：
```bash
python migrate_config.py
```

2. 验证新配置：
```bash
python config_manager.py validate conf_migrated.yaml
```

3. 切换到新配置：
```bash
python config_manager.py switch conf_migrated.yaml
```

## 配置迁移

### 自动迁移

使用提供的迁移脚本将旧配置转换为新格式：

```bash
# 基本迁移
python migrate_config.py

# 指定输入和输出文件
python migrate_config.py --input conf.yaml --output conf_migrated.yaml

# 备份原文件
python migrate_config.py --backup
```

### 手动迁移步骤

1. **备份当前配置**
   ```bash
   cp conf.yaml conf.yaml.backup
   ```

2. **使用新配置格式**
   ```bash
   cp conf_new.yaml conf.yaml
   ```

3. **根据需要调整配置**
   编辑 `conf.yaml` 文件，调整具体的配置值。

## 配置管理

### 使用配置管理脚本

```bash
# 列出所有配置文件
python config_manager.py list

# 验证配置文件
python config_manager.py validate conf_new.yaml

# 切换默认配置
python config_manager.py switch conf_new.yaml

# 比较两个配置文件
python config_manager.py compare conf.yaml conf_new.yaml

# 显示配置详细信息
python config_manager.py info conf_new.yaml
```

## 配置项说明

### 核心设置
- `report_style`: 报告样式 (academic, business, technical, casual)
- `resources`: 自定义资源列表

### LLM 配置
- `llm.temperature`: 模型温度 (0.0-1.0)
- `llm.timeout`: 请求超时时间（秒）
- `llm.max_tokens`: 最大令牌数

### 代理配置
- `agents.max_plan_iterations`: 最大计划迭代次数
- `agents.max_step_num`: 最大步骤数
- `agents.max_search_results`: 最大搜索结果数
- `agents.enable_deep_thinking`: 启用深度思考
- `agents.enable_parallel_execution`: 启用并行执行

### 研究配置
- `research.enable_researcher_isolation`: 启用研究员隔离
- `research.researcher_isolation_level`: 隔离级别 (minimal, moderate, aggressive)
- `research.researcher_max_local_context`: 最大本地上下文大小

### 反思配置
- `reflection.enable_enhanced_reflection`: 启用增强反思
- `reflection.max_reflection_loops`: 最大反思循环次数
- `reflection.reflection_temperature`: 反思温度

### 内容处理配置
- `content.enable_content_summarization`: 启用内容摘要
- `content.enable_smart_filtering`: 启用智能过滤
- `content.summary_type`: 摘要类型 (comprehensive, key_points, abstract)

### 高级上下文管理
- `advanced_context.max_context_ratio`: 最大上下文比例
- `advanced_context.sliding_window_size`: 滑动窗口大小
- `advanced_context.compression_threshold`: 压缩阈值

### MCP 配置
- `mcp.enabled`: 启用 MCP
- `mcp.servers`: MCP 服务器列表
- `mcp.timeout`: MCP 超时时间

### 模型令牌限制
- `model_token_limits`: 各模型的令牌限制配置

### 搜索引擎配置
- `search.engine`: 搜索引擎 (tavily, duckduckgo, brave)
- `search.timeout`: 搜索超时时间
- `search.max_retries`: 最大重试次数

### 性能配置
- `performance.enable_caching`: 启用缓存
- `performance.cache_size`: 缓存大小
- `performance.max_workers`: 最大工作线程数

## 环境变量支持

新配置系统支持通过环境变量覆盖配置：

```bash
# 设置 LLM 温度
export DEER_LLM_TEMPERATURE=0.8

# 设置最大搜索结果
export DEER_AGENTS_MAX_SEARCH_RESULTS=10

# 启用深度思考
export DEER_AGENTS_ENABLE_DEEP_THINKING=true

# 设置搜索引擎
export SELECTED_SEARCH_ENGINE=tavily
```

## 使用新配置启动系统

```bash
# 使用默认配置 (conf.yaml)
python main.py "你的问题"

# 使用指定配置文件
python main.py --config conf_new.yaml "你的问题"

# 交互模式
python main.py --interactive --config conf_new.yaml
```

## 配置验证

系统会自动验证配置文件的正确性：

- **类型检查**: 确保配置值的类型正确
- **必需字段**: 检查必需的配置项
- **值范围**: 验证数值在合理范围内
- **依赖关系**: 检查配置项之间的依赖关系

## 故障排除

### 常见问题

1. **配置文件不存在**
   ```
   错误: 配置文件 conf.yaml 不存在
   解决: cp conf.yaml.example conf.yaml
   ```

2. **配置格式错误**
   ```
   错误: YAML 解析失败
   解决: 检查 YAML 语法，确保缩进正确
   ```

3. **配置验证失败**
   ```
   错误: 配置验证失败
   解决: 使用 config_manager.py validate 检查配置
   ```

4. **环境变量未生效**
   ```
   错误: 环境变量设置无效
   解决: 确保环境变量名称正确，使用 DEER_ 前缀
   ```

### 调试配置

```bash
# 启用调试模式查看配置加载过程
python main.py --debug --config conf_new.yaml "测试问题"

# 验证配置文件
python config_manager.py validate conf_new.yaml

# 查看配置详细信息
python config_manager.py info conf_new.yaml
```

## 最佳实践

1. **使用新配置格式**: 推荐使用 `conf_new.yaml` 格式
2. **环境变量管理**: 敏感信息使用环境变量
3. **配置验证**: 修改配置后及时验证
4. **版本控制**: 将配置文件加入版本控制（排除敏感信息）
5. **文档更新**: 配置变更时更新相关文档

## 迁移建议

为了解决当前配置混乱的问题，建议按以下步骤进行：

1. **立即迁移到新格式**
   ```bash
   python migrate_config.py --backup
   python config_manager.py switch conf_migrated.yaml
   ```

2. **验证迁移结果**
   ```bash
   python config_manager.py validate conf.yaml
   python main.py --debug "测试问题"
   ```

3. **清理旧配置**
   - 保留 `conf.yaml.example` 作为参考
   - 删除或重命名旧的配置文件
   - 更新文档和脚本中的配置引用

4. **团队同步**
   - 通知团队成员配置格式变更
   - 更新部署脚本和 CI/CD 配置
   - 提供迁移指南和工具