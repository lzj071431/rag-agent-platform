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
## 🚀 快速开始
### 1. 环境准备
确保你已安装 Python 3.10+，并在项目目录下创建了虚拟环境：
```bash
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# Mac/Linux 激活  