# AgentFlow

AgentFlow 是一个基于 LangGraph 构建的多智能体问答系统。它通过路由节点把用户问题分发给不同职责的智能体，在文档检索、联网搜索、研究分析和最终回答之间做动态切换，适合用来演示更接近生产形态的 Agent 工作流。

当前项目提供了一个基于 Streamlit 的交互界面，支持：

- 普通对话问答
- 上传 PDF 后进行文档问答
- 联网搜索新闻与网页信息
- 查询天气与股票信息
- 使用 LangSmith 进行链路追踪与调试

## 项目特点

- 多智能体协作：把路由、检索、搜索、研究和回答拆分为独立角色
- 图工作流编排：基于 LangGraph 定义状态流转和条件分支
- 文档问答能力：上传 PDF 后自动抽取内容并写入本地知识库
- 工具增强：可按需调用网页搜索、新闻搜索、天气和股票工具
- 可观测性：支持接入 LangSmith 查看执行轨迹和调用细节
- 可扩展：可以继续增加新的 Agent、工具或业务流程

## 工作流说明

系统中的主要角色如下：

1. `Router Agent`
   负责理解用户问题，并决定后续交给哪个节点处理。

2. `RAG Agent`
   当用户问题与已上传文档相关时，使用本地知识库进行检索并返回上下文。

3. `Web Agent`
   负责处理需要联网获取信息的问题，例如最新新闻、实时信息等。

4. `Research Agent`
   用于执行更复杂的多步研究任务，组织中间过程并补充上下文。

5. `Answer Agent`
   汇总已有上下文，输出最终回复。

## 技术栈

- `LangGraph`：多智能体状态流与条件路由
- `LangChain`：模型、工具与提示词封装
- `LangSmith`：链路追踪、调试与评估
- `Streamlit`：Web 交互界面
- `OpenAI-Compatible API`：兼容 OpenAI 格式的模型接入
- `Tavily`：网页搜索
- `Google Serper`：新闻搜索
- `OpenWeatherMap`：天气查询
- `Alpha Vantage`：股票行情查询

## 项目结构

```text
AgentFlow/
├─ app.py                     # Streamlit 应用入口
├─ src/
│  ├─ agent/                 # Agent 编排、模型工厂、提示词
│  ├─ tools/                 # 外部工具封装
│  ├─ pinecone/              # 本地知识库读写与检索逻辑
│  ├─ logger/                # 日志模块
│  └─ exception/             # 异常处理
├─ utils/                    # 公共工具函数
├─ data/                     # 本地知识库存储目录
├─ assets/                   # 静态资源目录
├─ requirements.txt
└─ .env.example
```

说明：虽然目录名中保留了 `pinecone`，但当前代码实际使用的是本地 JSON 知识库存储，不依赖 Pinecone 在线向量数据库。

## 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/heizguang/agentflow-langgraph-orchestrator
cd agentflow-langgraph-orchestrator
```

### 2. 创建虚拟环境

使用 `conda`：

```bash
conda create -p venv python=3.11.5 -y
conda activate venv
```

或使用 `venv`：

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example`，并按需填写你的密钥和模型配置。

需要重点配置的变量包括：

```env
OPENAI_BASE_URL=
OPENAI_API_KEY=
OPENAI_MODEL=

TAVILY_API_KEY=
SERPER_API_KEY=
WEATHER_API_KEY=
STOCK_FINANCE_API_KEY=

LANGSMITH_API_KEY=
LANGCHAIN_PROJECT=AgentFlow
```

如果你希望为不同 Agent 指定不同模型，还可以单独配置：

```env
AGENTFLOW_MODEL=
AGENTFLOW_ROUTER_MODEL=
AGENTFLOW_RAG_MODEL=
AGENTFLOW_WEB_MODEL=
AGENTFLOW_ANSWER_MODEL=
AGENTFLOW_RESEARCH_MODEL=
```

### 5. 启动应用

```bash
streamlit run app.py
```

启动后在浏览器打开本地地址：

```text
http://localhost:8501/
```

## 使用方式

### 普通问答

直接在聊天框输入问题，系统会自动判断是直接回答、联网搜索还是进入研究流程。

### 文档问答

1. 在左侧边栏上传 PDF 文件
2. 系统会自动解析文本并写入本地知识库
3. 上传完成后，可以围绕文档内容继续提问

### 历史会话

左侧边栏会显示已有会话线程，可以切换查看历史对话内容。

## 已知限制

- 依赖多个外部 API，未配置密钥时部分功能无法使用
- 免费额度接口通常存在速率限制
- 联网查询和多步研究会带来更高延迟
- 文档检索当前基于本地文本分块与简单召回，不是完整的高性能向量数据库方案

## 后续可扩展方向

- 增加更多垂直 Agent，例如邮件、日程、任务自动化
- 接入更多工具平台，例如 Gmail、Slack、Notion
- 引入更强的记忆机制，支持跨会话个性化
- 将 Streamlit 界面替换为更适合生产部署的后端 API 架构
- 优化本地知识库检索质量与文档处理流程

## 维护说明

本项目由仓库维护者与贡献者共同维护。
