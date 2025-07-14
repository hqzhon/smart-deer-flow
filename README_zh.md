<div align="center">

# 🦌 SmartDeerFlow

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hqzhon/smart-deer-flow)

**AI 驱动的深度研究框架，支持多智能体协作**

[English](./README.md) | [简体中文](./README_zh.md)

</div>

## 🚀 概述

**SmartDeerFlow** 是一个社区驱动的 AI 研究框架，结合了**大型语言模型**、**多智能体系统**和**高级工具**，用于自动化研究、内容生成和数据分析。

**核心亮点：**
- 🤖 **多智能体协作** - 基于角色专业化的智能任务分配和跨智能体通信
- 🧠 **GFLQ 反射系统** - 通过知识缺口检测和迭代增强实现自我改进的研究质量
- 🔄 **自适应研究流程** - 多智能体间的动态策略调整和共识构建
- 🔍 **多源智能集成** - Tavily、Brave、DuckDuckGo、ArXiv 集成与智能内容分析
- 📊 **丰富输出格式** - AI 驱动的报告、播客、演示文稿生成
- 🌐 **灵活交互界面** - Web 和控制台 UI，支持人机协作循环
- 🔗 **可扩展架构** - MCP 协议集成和 RAG 知识库支持

> 基于 [DeerFlow](https://github.com/bytedance/deer-flow) 开发，增加了增强功能和社区驱动的改进。

---

## 📑 目录

- [🚀 快速开始](#-快速开始)
- [🌟 功能特性](#-功能特性)
- [⚡ 性能](#-性能)
- [🏗️ 架构](#-架构)
- [🔄 GFLQ 反射集成](#-gflq-反射集成)
- [📚 示例](#-示例)
- [🐳 Docker](#-docker)
- [🛠️ 开发](#️-开发)
- [❓ 常见问题](#-常见问题)
- [📜 许可证](#-许可证)

## 🚀 快速开始

### 先决条件
- **Python 3.12+** 和 **Node.js 22+**
- 推荐工具：[`uv`](https://docs.astral.sh/uv/) (Python) 和 [`pnpm`](https://pnpm.io/) (Node.js)

### 安装

#### 1. 克隆和设置
```bash
git clone https://github.com/hqzhon/smart-deer-flow.git
cd smart-deer-flow
uv sync  # 自动创建虚拟环境并安装依赖
```

#### 2. 配置 API 密钥
```bash
cp .env.example .env
cp conf.yaml.example conf.yaml
# 编辑 .env 和 conf.yaml 文件，添加您的 API 密钥
```

#### 3. 可选：安装额外工具
```bash
# Web UI 依赖
cd web && pnpm install

# PPT 生成工具
brew install marp-cli
```

### 使用方法

#### 控制台模式（快速启动）
```bash
uv run main.py "您的研究问题"
```

#### 交互模式
```bash
uv run main.py --interactive
```

#### Web UI 模式
```bash
./bootstrap.sh -d  # macOS/Linux
bootstrap.bat -d   # Windows
# 访问 http://localhost:3000
```

📖 **配置指南：** [详细配置说明](docs/configuration_guide.md)

## 支持的搜索引擎

DeerFlow 支持多种搜索引擎，可以在`.env`文件中通过`SEARCH_API`变量进行配置：

- **Tavily**（默认）：专为 AI 应用设计的专业搜索 API

  - 需要在`.env`文件中设置`TAVILY_API_KEY`
  - 注册地址：https://app.tavily.com/home

- **DuckDuckGo**：注重隐私的搜索引擎

  - 无需 API 密钥

- **Brave Search**：具有高级功能的注重隐私的搜索引擎

  - 需要在`.env`文件中设置`BRAVE_SEARCH_API_KEY`
  - 注册地址：https://brave.com/search/api/

- **Arxiv**：用于学术研究的科学论文搜索
  - 无需 API 密钥
  - 专为科学和学术论文设计

要配置您首选的搜索引擎，请在`.env`文件中设置`SEARCH_API`变量：

```bash
# 选择一个：tavily, duckduckgo, brave_search, arxiv
SEARCH_API=tavily
```

## 🌟 功能特性

### 🤖 AI 与 LLM 集成
- **多智能体协作** - 基于 LangGraph 的专业化智能体系统
- **自适应研究策略** - 根据发现动态调整研究方向
- **智能内容生成** - AI 驱动的高质量内容创作
- **MCP 服务器支持** - 扩展模型上下文协议集成

### 🔍 研究与数据收集
- **多源搜索引擎** - Tavily、Brave、DuckDuckGo、ArXiv 集成
- **并行信息检索** - 同时查询多个数据源
- **智能过滤与排序** - 高质量信息筛选
- **实时数据更新** - 获取最新信息

### 🤖 多智能体协作
- **智能任务分配** - 协调器管理工作流程
- **专业化智能体** - 研究员、编码员、报告员
- **反射机制** - GFLQ 质量评估和改进
- **状态管理** - 基于 LangGraph 的工作流

### 📊 内容生成
- **多格式输出** - Markdown、PDF、DOCX、PowerPoint
- **模板系统** - 可自定义输出格式
- **数据可视化** - 图表和图形生成
- **引用管理** - 自动引用和参考文献

### 👥 人机协作
- **交互模式** - 实时协作和反馈
- **人工干预** - 生成过程中的编辑控制
- **质量审核** - 人工验证机制
- **迭代优化** - 基于反馈的持续改进

### 🔗 MCP 集成
- **模型上下文协议** - 扩展 AI 能力
- **工具生态系统** - 丰富的集成选项
- **可扩展架构** - 支持自定义工具
- **标准化接口** - 一致的集成体验

### 🧠 高级 AI 功能
- **反射机制** - 自我评估和改进
- **知识缺口检测** - 识别信息不足
- **动态查询生成** - 智能补充研究
- **质量保证** - 持续优化输出

## ⚡ 性能

### 智能体协调工具
- **并行处理** - 多智能体同时执行任务
- **负载均衡** - 智能分配计算资源
- **缓存机制** - 提升响应速度和效率
- **状态管理** - 优化工作流程执行

### 智能体配置

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `AGENT_COORDINATION_ENABLED` | 启用智能体协调 | `true` |
| `MAX_CONCURRENT_AGENTS` | 最大并发智能体数 | `4` |
| `REFLECTION_ENABLED` | 启用反射机制 | `false` |
| `CACHE_ENABLED` | 启用缓存 | `true` |

### 智能体模式

| 模式 | 智能体数 | 特性 |
|------|----------|------|
| **Basic** | 2-3 | 基础协作 |
| **Standard** | 4-6 | 标准多智能体 |
| **Collaborative** | 6-8 | 反射机制 |
| **Advanced** | 8+ | 全功能优化 |

```bash
# 启用高级智能体模式
uv run main.py "研究主题" --agent-mode collaborative

# 配置环境变量
export REFLECTION_ENABLED=true
export MAX_CONCURRENT_AGENTS=6
```

## 🏗️ 架构

SmartDeerFlow 采用先进的多智能体架构，每个智能体都有专门的职责和优化的工具集。系统基于 LangGraph 构建，实现了灵活的状态管理和智能体间协作。

![多智能体架构](https://mdn.alipayobjects.com/one_clip/afts/img/EyXoT63Uq1QAAAAASNAAAAgAoEACAQFr/original)

### 🤖 核心智能体

**协调器智能体** - 系统的入口点和任务管理器
- 接收用户输入并启动研究流程
- 管理整体工作流程和智能体协调
- 作为用户与系统的主要交互界面

**规划器智能体** - 战略规划和任务分解
- 分析研究目标并制定执行计划
- 评估上下文充分性和研究需求
- 决定何时生成最终报告

**研究节点** - 信息收集和数据检索
- 使用多种搜索引擎进行网络搜索
- 集成 MCP 服务扩展研究能力
- 执行并行信息检索和过滤

**编码器智能体** - 技术分析和代码执行
- 使用 Python REPL 进行代码分析
- 处理技术任务和数据处理
- 提供代码执行和验证能力

**报告器智能体** - 内容生成和格式化
- 汇总和组织研究发现
- 生成多格式输出（Markdown、PDF、DOCX）
- 管理引用和参考文献

### 🔄 反射机制集成

系统集成了 GFLQ 反射功能，通过智能反馈循环提升研究质量：

1. **反射分析** - 评估研究结果的质量和完整性
2. **知识缺口检测** - 识别需要进一步研究的领域
3. **后续研究生成** - 基于缺口分析创建补充查询
4. **迭代优化** - 持续改进直到达到质量标准



## 🔄 GFLQ 反射集成

### 概述

GFLQ（生成-反馈-学习-查询）反射机制是一个实验性的AI增强功能，旨在通过自我反思和迭代改进来提升研究质量。该系统允许AI智能体在研究过程中进行自我评估和优化。

### 关键特性

- **🔄 自我反思机制** - AI智能体能够评估自己的输出质量
- **📈 迭代改进** - 基于反思结果自动优化研究策略
- **🎯 质量控制** - 实时监控和调整研究方向
- **⚡ 快速验证** - 实验性功能，支持快速开启/关闭

### 带来的好处

- **🎯 提升研究质量** - 通过自我反思机制提高输出准确性
- **🔄 自适应优化** - 根据反馈自动调整研究策略
- **⚡ 实验性验证** - 快速验证新的AI增强技术
- **📊 质量监控** - 实时跟踪研究质量指标

> 📖 **详细文档：** 查看 [GFLQ 反射机制](./docs/GFLQ_REFLECTION_zh.md) 了解完整的技术细节。

## 🛠️ 开发

### 快速命令
```bash
# 测试
uv run pytest --cov=deer_flow

# 代码质量
uv run ruff check --fix && uv run ruff format

# LangGraph Studio 调试
langgraph dev  # 访问 http://localhost:2024
```

## 🐳 Docker

### 快速启动
```bash
docker-compose up -d
# 访问 http://localhost:3000
```

**包含服务：** Web UI、API 后端、Redis 缓存

### LangGraph Studio 调试

DeerFlow 使用 LangGraph 作为其工作流架构。您可以使用 LangGraph Studio 实时调试和可视化工作流。

#### 本地运行 LangGraph Studio

DeerFlow 包含一个`langgraph.json`配置文件，该文件定义了 LangGraph Studio 的图结构和依赖关系。该文件指向项目中定义的工作流图，并自动从`.env`文件加载环境变量。

##### Mac

```bash
# 如果您没有uv包管理器，请安装它
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖并启动LangGraph服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
```

##### Windows / Linux

```bash
# 安装依赖
pip install -e .
pip install -U "langgraph-cli[inmem]"

# 启动LangGraph服务器
langgraph dev
```

启动 LangGraph 服务器后，您将在终端中看到几个 URL：

- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API 文档: http://127.0.0.1:2024/docs

在浏览器中打开 Studio UI 链接以访问调试界面。

#### 使用 LangGraph Studio

在 Studio UI 中，您可以：

1. 可视化工作流图并查看组件如何连接
2. 实时跟踪执行情况，了解数据如何在系统中流动
3. 检查工作流每个步骤的状态
4. 通过检查每个组件的输入和输出来调试问题
5. 在规划阶段提供反馈以完善研究计划

当您在 Studio UI 中提交研究主题时，您将能够看到整个工作流执行过程，包括：

- 创建研究计划的规划阶段
- 可以修改计划的反馈循环
- 每个部分的研究和写作阶段
- 最终报告生成

### 启用 LangSmith 追踪

DeerFlow 支持 LangSmith 追踪功能，帮助您调试和监控工作流。要启用 LangSmith 追踪：

1. 确保您的 `.env` 文件中有以下配置（参见 `.env.example`）：
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   LANGSMITH_API_KEY="xxx"
   LANGSMITH_PROJECT="xxx"
   ```

2. 通过运行以下命令本地启动 LangSmith 追踪：
   ```bash
   langgraph dev
   ```

这将在 LangGraph Studio 中启用追踪可视化，并将您的追踪发送到 LangSmith 进行监控和分析。

## Docker

您也可以使用 Docker 运行此项目。

首先，您需要阅读下面的[配置](#配置)部分。确保`.env`和`.conf.yaml`文件已准备就绪。

其次，构建您自己的 Web 服务器 Docker 镜像：

```bash
docker build -t deer-flow-api .
```

最后，启动运行 Web 服务器的 Docker 容器：

```bash
# 将deer-flow-api-app替换为您首选的容器名称
docker run -d -t -p 8000:8000 --env-file .env --name deer-flow-api-app deer-flow-api

# 停止服务器
docker stop deer-flow-api-app
```

### Docker Compose

您也可以使用 docker compose 设置此项目：

```bash
# 构建docker镜像
docker compose build

# 启动服务器
docker compose up
```

## 文本转语音集成

DeerFlow 现在包含一个文本转语音(TTS)功能，允许您将研究报告转换为语音。此功能使用火山引擎 TTS API 生成高质量的文本音频。速度、音量和音调等特性也可以自定义。

### 使用 TTS API

您可以通过`/api/tts`端点访问 TTS 功能：

```bash
# 使用curl的API调用示例
curl --location 'http://localhost:8000/api/tts' \
--header 'Content-Type: application/json' \
--data '{
    "text": "这是文本转语音功能的测试。",
    "speed_ratio": 1.0,
    "volume_ratio": 1.0,
    "pitch_ratio": 1.0
}' \
--output speech.mp3
```

## 📚 示例

### 研究报告
```bash
# 基础研究报告
uv run main.py "人工智能在医疗保健中的影响"

# 启用反射机制的深度研究
uv run main.py "量子计算的未来发展" --agent-mode collaborative

# 交互式研究会话
uv run main.py --interactive
```

### 示例报告
- [AI 在医疗保健中的应用](./examples/healthcare_ai_report.md)
- [量子计算技术分析](./examples/quantum_computing_analysis.md)
- [市场趋势研究](./examples/market_trends_study.md)

## 🔧 配置与使用

### 命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--interactive` | 启用交互模式 | `false` |
| `--agent-mode` | 智能体模式 (basic/standard/collaborative/advanced) | `standard` |
| `--max-plan-iterations` | 最大规划迭代次数 | `3` |
| `--human-in-the-loop` | 启用人工干预 | `false` |
| `--output-format` | 输出格式 (markdown/pdf/docx/ppt) | `markdown` |

### 交互模式

应用程序现在支持带有英文和中文内置问题的交互模式：

1. 启动交互模式：

   ```bash
   uv run main.py --interactive
   ```

2. 选择您偏好的语言（English 或中文）

3. 从内置问题列表中选择或选择提出您自己问题的选项

4. 系统将处理您的问题并生成全面的研究报告

### 人在环中

DeerFlow 包含一个人在环中机制，允许您在执行研究计划前审查、编辑和批准：

1. **计划审查**：启用人在环中时，系统将在执行前向您展示生成的研究计划

2. **提供反馈**：您可以：

   - 通过回复`[ACCEPTED]`接受计划
   - 通过提供反馈编辑计划（例如，`[EDIT PLAN] 添加更多关于技术实现的步骤`）
   - 系统将整合您的反馈并生成修订后的计划

3. **自动接受**：您可以启用自动接受以跳过审查过程：

   - 通过 API：在请求中设置`auto_accepted_plan: true`

4. **API 集成**：使用 API 时，您可以通过`feedback`参数提供反馈：
   ```json
   {
     "messages": [{ "role": "user", "content": "什么是量子计算？" }],
     "thread_id": "my_thread_id",
     "auto_accepted_plan": false,
     "feedback": "[EDIT PLAN] 包含更多关于量子算法的内容"
   }
   ```

### 命令行参数

应用程序支持多个命令行参数来自定义其行为：

- **query**：要处理的研究查询（可以是多个词）
- **--interactive**：以交互模式运行，带有内置问题
- **--max_plan_iterations**：最大规划周期数（默认：1）
- **--max_step_num**：研究计划中的最大步骤数（默认：3）
- **--debug**：启用详细调试日志

## ❓ 常见问题

**Q: 如何配置 API 密钥？**  
A: 复制 `.env.example` 到 `.env`，然后添加您的 API 密钥（OpenAI、Tavily 等）。

**Q: 可以使用本地模型吗？**  
A: 支持！通过 Ollama、vLLM、LM Studio 等本地推理服务。

**Q: 如何启用反射机制？**  
A: 在 `conf.yaml` 中设置 `gflq_reflection.enabled: true` 并配置相关参数。

**Q: 支持哪些搜索引擎？**  
A: Tavily（推荐）、Brave、DuckDuckGo、ArXiv 等多种搜索引擎。

**Q: 如何贡献代码？**  
A: Fork 仓库 → 创建分支 → 提交更改 → 发起 Pull Request。

## 📜 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

**基于 [DeerFlow](https://github.com/bytedance/deer-flow) 开发** | **核心技术：** LangChain/LangGraph、多智能体系统、反射机制

**反射功能**参考[gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)
