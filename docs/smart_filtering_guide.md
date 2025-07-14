# 智能筛选功能使用指南

## 概述

智能筛选功能是DeerFlow系统中的一个重要特性，旨在解决网络搜索返回信息量过大导致的token超限问题。该功能通过LLM模型对搜索结果进行智能筛选和合并，只保留与用户查询最相关的信息。

## 功能特点

### 🎯 智能相关性筛选
- 使用LLM模型分析搜索结果与查询的相关性
- 自动过滤无关或低质量内容
- 保留最有价值的信息片段

### 📊 分批处理机制
- 支持大量搜索结果的分批处理
- 避免单次处理token超限
- 提高处理效率和稳定性

### 🔄 二次筛选优化
- 当初次筛选结果仍然过长时，自动进行二次筛选
- 进一步精炼内容，确保符合token限制
- 保持信息的完整性和准确性

### ⚡ 智能回退机制
- 当智能筛选失败时，自动回退到传统处理方式
- 确保系统的稳定性和可靠性
- 提供多层次的容错保护

## 配置说明

### 1. 配置文件设置

在 `conf.yaml` 文件中添加以下配置：

```yaml
CONTENT_PROCESSING:
  enable_smart_chunking: true          # 启用智能分块
  enable_content_summarization: true   # 启用内容摘要
  enable_smart_filtering: true         # 启用智能筛选 (新增)
  chunk_strategy: "auto"               # 分块策略
  summary_type: "comprehensive"        # 摘要类型
```

### 2. 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_smart_filtering` | boolean | `true` | 是否启用LLM智能筛选功能 |
| `enable_smart_chunking` | boolean | `true` | 是否启用智能分块（回退选项） |
| `enable_content_summarization` | boolean | `true` | 是否启用内容摘要（回退选项） |

## 工作流程

### 1. 搜索结果获取
```
用户查询 → 搜索引擎 → 原始搜索结果
```

### 2. 智能筛选处理
```
原始结果 → Token限制检查 → 智能筛选 → 筛选结果
```

### 3. 回退机制
```
筛选失败 → 智能摘要 → 智能分块 → 最终结果
```

## 核心组件

### SearchResultFilter 类

位置：`src/utils/search_result_filter.py`

主要方法：
- `filter_search_results()`: 主要筛选方法
- `_create_filter_prompt()`: 创建筛选提示
- `_parse_filter_response()`: 解析筛选响应
- `_format_filtered_results()`: 格式化筛选结果

### ContentProcessor 集成

位置：`src/utils/content_processor.py`

增强的 `process_search_results()` 方法现在支持：
- `query` 参数：用户查询内容
- `enable_smart_filtering` 参数：是否启用智能筛选

## 使用示例

### 1. 基本使用

```python
from src.utils.common.search_result_filter import SearchResultFilter
from src.utils.tokens.content_processor import ContentProcessor

# 创建内容处理器和筛选器实例
processor = ContentProcessor()
filter_instance = SearchResultFilter(processor)

# 执行筛选（不再需要 LLM 参数）
filtered_results = filter_instance.filter_search_results(
    query="Python机器学习教程",
    search_results=search_results,
    model_name="deepseek-chat",
    max_results=3
)
```

### 2. 集成到内容处理器

```python
from src.utils.tokens.content_processor import ContentProcessor
from src.utils.common.search_result_filter import SearchResultFilter

# 创建处理器和筛选器实例
processor = ContentProcessor()
filter_instance = SearchResultFilter(processor)

# 处理搜索结果（使用新的轻量级筛选）
filtered_results = filter_instance.filter_search_results(
    query="深度学习框架对比",
    search_results=search_results,
    model_name="deepseek-chat",
    max_results=5
)
```

## 性能优化

### 1. 批处理大小
- 默认批处理大小：5个结果
- 可根据模型token限制调整
- 平衡处理效率和质量

### 2. 相关性阈值
- 默认相关性阈值：7.0（满分10分）
- 可根据应用场景调整
- 更高阈值 = 更严格筛选

### 3. 二次筛选触发
- 当筛选结果超过token限制的80%时触发
- 进一步压缩内容长度
- 保持信息质量

## 监控和日志

### 日志级别
- `INFO`: 筛选过程关键信息
- `DEBUG`: 详细的筛选步骤
- `WARNING`: 筛选失败或异常
- `ERROR`: 严重错误和异常

### 关键指标
- 筛选前后结果数量变化
- 内容长度压缩比例
- 筛选成功率
- 处理时间

## 故障排除

### 常见问题

1. **筛选结果为空**
   - 检查相关性阈值设置
   - 确认查询与搜索结果的匹配度
   - 查看LLM响应日志

2. **筛选失败频繁**
   - 检查LLM模型配置
   - 确认网络连接稳定性
   - 查看错误日志详情

3. **性能问题**
   - 调整批处理大小
   - 优化相关性阈值
   - 考虑使用更快的LLM模型

### 调试模式

启用详细日志：
```python
import logging
logging.getLogger('src.utils.search_result_filter').setLevel(logging.DEBUG)
```

## 测试验证

运行测试脚本验证功能：

```bash
python test_smart_filtering.py
```

测试内容包括：
- 基本筛选功能
- 内容处理器集成
- 错误处理机制
- 性能基准测试

## 最佳实践

### 1. 配置建议
- 生产环境建议启用所有智能处理功能
- 根据实际token限制调整相关参数
- 定期监控筛选效果和性能

### 2. 查询优化
- 使用具体、明确的查询词
- 避免过于宽泛的查询
- 考虑查询的语言和领域特点

### 3. 结果验证
- 定期检查筛选结果的质量
- 收集用户反馈进行优化
- 监控相关性评分分布

## 未来改进

### 计划中的功能
- 支持自定义筛选策略
- 增加多语言筛选支持
- 实现筛选结果缓存机制
- 添加筛选效果评估工具

### 性能优化
- 并行处理多个批次
- 智能批处理大小调整
- 缓存常见查询的筛选结果

---

## 技术支持

如有问题或建议，请：
1. 查看日志文件获取详细错误信息
2. 运行测试脚本验证功能状态
3. 检查配置文件设置是否正确
4. 确认LLM模型可用性和配置