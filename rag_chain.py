"""向后兼容层 —— 旧版 ``rag_chain`` 模块。

实际实现已迁移到 :mod:`services.retrieval_service` 与 :mod:`services.rag_service`。
"""

from __future__ import annotations

from services.rag_service import RAGAnswer, RAGService, build_rag_chain, build_rag_service
from services.retrieval_service import (
    HybridRRFRetriever,
    build_hybrid_retriever,
    format_docs,
)
from utils.bm25 import SimpleBM25

#: 旧代码可能引用该常量，保留别名
PROMPT_TEMPLATE = __import__("core.prompts", fromlist=["RAG_QA_TEMPLATE"]).RAG_QA_TEMPLATE

__all__ = [
    "PROMPT_TEMPLATE",
    "HybridRRFRetriever",
    "SimpleBM25",
    "build_hybrid_retriever",
    "build_rag_chain",
    "build_rag_service",
    "format_docs",
    "RAGService",
    "RAGAnswer",
]
