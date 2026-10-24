# 📄 基于 LangChain + DeepSeek 的 RAG 财报智能问答系统

## 📖 项目简介
这是一个基于 **LangChain 1.0 + DeepSeek + ChromaDB** 构建的检索增强生成（RAG）问答系统。
用户只需上传一份财务报告 PDF，系统便会自动解析、切分并向量化。用户提问时，系统会从文档中检索相关片段，并由 DeepSeek 生成精准、可溯源的回答。

## 🏗️ 系统架构
本项目的数据流转如下：
1. **文档加载**：`PyPDFLoader` 解析 PDF 文本。
2. **文本清洗与切分**：`RecursiveCharacterTextSplitter`（chunk_size=500, overlap=50），加入正则清洗去除页眉页脚与目录噪声。
3. **向量化存储**：使用 `BAAI/bge-small-zh-v1.5` 生成 Embedding，存入 `ChromaDB`。
4. **检索增强**：基础向量检索（Top-10），并可选集成 `BAAI/bge-reranker-base` 进行重排序，提升准确率。
5. **生成模型**：调用 DeepSeek (`deepseek-chat`) 生成最终回答。
6. **前端界面**：基于 `Streamlit` 构建，支持文件上传、构建向量库、交互问答与来源引用展示。

## 📊 评估数据
为验证系统效果，我自建了 **20 道题的黄金测试集**（包含事实型、数值型、否定型等），评估结果如下：

| 指标 | 基线 (纯向量检索) | 优化 (引入Reranker) |
| :--- | :--- | :--- |
| Hit Rate@5 | 95.00% | 95.00% |
| MRR | 0.7583 | 0.7583 |

> **💡 结论与反思**：在基线 Hit Rate 高达 95% 的情况下，Reranker 的优化效果边际递减。引入 Reranker 需权衡其响应延迟和部署成本（模型体积大、内存占用高）。在当前中小规模数据场景下，基础向量检索已足以满足需求。Reranker 更适合处理复杂跨段落问答，我已将其作为后续优化方向保留。

## 🏛️ 工程架构

代码按职责分层，`app.py` 只做 UI 编排，不含业务逻辑：

```
app.py              # Streamlit UI 入口（仅页面编排）
core/               # 基础设施：config(Pydantic Settings) / logger / exceptions / prompts
utils/              # 通用工具：retry(Tenacity) / bm25 / text_cleaner / webhook
services/           # 业务逻辑：RAG / 检索 / 向量库 / 周报 / CrewAI Agent
```

**核心特性**：

| 能力 | 实现 |
| :--- | :--- |
| 配置管理 | `core/config.py`，Pydantic Settings 统一校验，无散落 `os.getenv()` |
| 容错重试 | `utils/retry.py`，指数退避 2s→4s→8s，仅重试超时/5xx/429 |
| 结构化日志 | `core/logger.py`，带时间戳与模块名，支持 text/JSON 双格式与轮转 |

根目录的 `config.py` / `loader.py` / `splitter.py` / `vector_store.py` /
`rag_chain.py` / `crewai_analysis.py` / `weekly_report_agent.py` 是**兼容转发层**，
旧脚本可继续直接使用。

## 📦 部署

`deploy/` 是**生成产物**，由 `sync_deploy.py` 从根目录同步，不要手工修改：

```bash
# 1. 同步代码到 deploy/（保留 deploy/.env，自动校验哈希）
.venv\Scripts\python.exe sync_deploy.py

# 2. 上传并重建容器
.venv\Scripts\python.exe auto_deploy_full.py
```

同步脚本的其它用法：

```bash
python sync_deploy.py --dry-run   # 预览将要复制/删除的文件
python sync_deploy.py --check     # 只校验是否已同步（退出码 2 表示有漂移，可用于 CI）
```

> **安全设计**：`sync_deploy.py` 使用显式白名单，`auto_deploy*.py`、`.deploy_pw`
> 等含服务器凭据的文件**绝不会**被复制进 `deploy/`。

## 🚀 快速开始
### 1. 环境准备
确保你已安装 Python 3.10+，并在项目目录下创建了虚拟环境：
```bash
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# Mac/Linux 激活  