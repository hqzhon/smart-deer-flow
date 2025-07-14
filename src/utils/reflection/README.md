# 多语言反射Prompt框架

本模块为DeerFlow的增强反射机制提供多语言Prompt模板和管理功能。

## 功能特性

- 🌍 **多语言支持**: 支持中文和英文两种语言
- 📝 **模板化设计**: 使用Jinja2模板引擎，支持变量替换
- 🔧 **灵活配置**: 可根据上下文自动选择语言
- 📊 **质量评估**: 提供多语言的质量评估标签
- 🎯 **智能查询**: 生成针对性的后续查询建议

## 文件结构

```
src/prompts/reflection/
├── __init__.py                    # 模块初始化
├── reflection_analysis.md          # 反射分析模板
├── reflection_prompt_manager.py    # Prompt管理器
└── README.md                      # 使用文档
```

## 使用方法

### 基本使用

```python
from src.utils.reflection.reflection_prompt_manager import ReflectionPromptManager
from src.report_quality.i18n import Language

# 创建Prompt管理器
manager = ReflectionPromptManager()

# 生成中文反射分析Prompt
zh_prompt = manager.get_reflection_analysis_prompt(
    research_topic="人工智能在医疗领域的应用",
    current_step_index=2,
    total_steps=5,
    current_reflection_loop=1,
    max_reflection_loops=3,
    steps_summary="已完成的研究步骤摘要",
    results_summary="执行结果摘要",
    observations_summary="当前观察摘要",
    language=Language.ZH_CN
)

# 生成英文反射分析Prompt
en_prompt = manager.get_reflection_analysis_prompt(
    research_topic="AI applications in healthcare",
    current_step_index=2,
    total_steps=5,
    current_reflection_loop=1,
    max_reflection_loops=3,
    steps_summary="Summary of completed research steps",
    results_summary="Summary of execution results",
    observations_summary="Summary of current observations",
    language=Language.EN_US
)
```

### 在反射代理中使用

```python
from src.utils.reflection.enhanced_reflection import EnhancedReflectionAgent, ReflectionContext
from src.report_quality.i18n import Language

# 创建反射代理
agent = EnhancedReflectionAgent()

# 创建中文研究上下文
context = ReflectionContext(
    research_topic="区块链技术发展趋势",
    completed_steps=[...],
    execution_results=[...],
    observations=[...],
    locale="zh-CN"  # 设置为中文
)

# 执行反射分析（自动使用中文Prompt）
result = await agent.analyze_knowledge_gaps(context)
```

### 后续查询生成

```python
# 生成中文后续查询
zh_queries = await agent.generate_follow_up_queries(
    research_topic="人工智能发展趋势",
    knowledge_gaps=["缺乏最新技术数据", "监管政策不明确"],
    priority_areas=["技术安全性", "商业应用"],
    language=Language.ZH_CN
)

# 生成英文后续查询
en_queries = await agent.generate_follow_up_queries(
    research_topic="AI development trends",
    knowledge_gaps=["Lack of latest technical data", "Unclear regulatory policies"],
    priority_areas=["Technical security", "Commercial applications"],
    language=Language.EN_US
)
```

## 模板变量

反射分析模板支持以下变量：

- `CURRENT_TIME`: 当前时间
- `locale`: 语言环境（zh-CN 或 en-US）
- `research_topic`: 研究主题
- `current_date`: 当前日期
- `current_step_index`: 当前步骤索引
- `total_steps`: 总步骤数
- `current_reflection_loop`: 当前反射循环
- `max_reflection_loops`: 最大反射循环数
- `steps_summary`: 已完成步骤摘要
- `results_summary`: 执行结果摘要
- `observations_summary`: 当前观察摘要

## 质量评估标签

```python
# 获取中文质量评估标签
zh_labels = manager.get_quality_assessment_labels(Language.ZH_CN)
# 输出: {'completeness': '完整性', 'accuracy': '准确性', ...}

# 获取英文质量评估标签
en_labels = manager.get_quality_assessment_labels(Language.EN_US)
# 输出: {'completeness': 'Completeness', 'accuracy': 'Accuracy', ...}
```

## 建议分类标签

```python
# 获取中文建议分类标签
zh_categories = manager.get_recommendation_categories(Language.ZH_CN)
# 输出: {'methodology': '方法论改进', 'sources': '信息源扩展', ...}

# 获取英文建议分类标签
en_categories = manager.get_recommendation_categories(Language.EN_US)
# 输出: {'methodology': 'Methodology Improvement', 'sources': 'Source Expansion', ...}
```

## 扩展支持

要添加新语言支持：

1. 在 `reflection_analysis.md` 模板中添加新语言的条件分支
2. 在 `ReflectionPromptManager` 中添加相应的标签翻译
3. 在 `Language` 枚举中添加新的语言代码

## 注意事项

- 语言选择基于 `ReflectionContext.locale` 字段自动判断
- 模板使用Jinja2语法，支持条件语句和循环
- 所有文本内容都经过本地化处理
- 建议在生产环境中缓存生成的Prompt以提高性能