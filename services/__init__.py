"""services —— 核心业务逻辑层。

UI（``app.py``）与 CLI 只依赖本层，不直接接触 LangChain / Chroma / CrewAI
等第三方库。这样替换底层实现（例如把 Streamlit 换成 FastAPI）时，
业务代码零改动。

模块划分
--------
* :mod:`services.embeddings`      —— Embedding 模型工厂（进程级单例）
* :mod:`services.llm_factory`     —— 大模型客户端工厂（含重试）
* :mod:`services.document_service`—— 文档加载与切分
* :mod:`services.vector_store_service` —— 向量库构建 / 加载 / 重置
* :mod:`services.retrieval_service`    —— 混合检索（向量 + BM25 + RRF）
* :mod:`services.rag_service`          —— RAG 问答编排
* :mod:`services.report_service`       —— 周报 LangGraph 工作流
* :mod:`services.agent_service`        —— CrewAI 竞品分析
"""
