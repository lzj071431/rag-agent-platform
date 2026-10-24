"""向后兼容层 —— 旧版 ``config`` 模块。

历史上业务代码大量使用 ``from config import PERSIST_DIR, MODEL_NAME``。
为保证既有脚本（``build_vectorstore.py`` / ``evaluate_retrieval.py`` /
``verify_retrieval.py`` 等）无需改动即可继续运行，这里把新配置中心
``core.config.settings`` 的字段以**旧的扁平常量名**重新导出。

新代码请直接使用::

    from core.config import settings
    settings.persist_dir
    settings.chunking.size

⚠️ 本文件仅做转发，不包含任何逻辑。
"""

from __future__ import annotations

# 先导入 core.config，确保离线开关在任何第三方库之前生效
from core.config import PROJECT_ROOT, get_settings, settings

# ================= 大模型 =================
DEEPSEEK_API_KEY: str = settings.deepseek_api_key.get_secret_value()
DEEPSEEK_BASE_URL: str = settings.deepseek_base_url
MODEL_NAME: str = settings.model_name
LLM_TEMPERATURE: float = settings.llm_temperature

# ================= 切分参数 =================
CHUNK_SIZE: int = settings.chunking.size
CHUNK_OVERLAP: int = settings.chunking.overlap
SECTION_MARKER: str = settings.chunking.section_marker

# ================= 检索参数 =================
VECTOR_K: int = settings.retrieval.vector_k
BM25_K: int = settings.retrieval.bm25_k
FINAL_K: int = settings.retrieval.final_k
RRF_C: int = settings.retrieval.rrf_c
BM25_WEIGHT: float = settings.retrieval.bm25_weight
ENABLE_HYBRID: bool = settings.retrieval.enable_hybrid
ENABLE_MULTI_QUERY: bool = settings.retrieval.enable_multi_query

# ================= 向量库 =================
EMBEDDING_MODEL: str = settings.embedding_model
PERSIST_DIR: str = settings.persist_dir
COLLECTION_NAME: str = settings.collection_name

# ================= 外部集成 =================
FEISHU_WEBHOOK_URL: str = settings.feishu_webhook_url.get_secret_value()

__all__ = [
    "PROJECT_ROOT",
    "get_settings",
    "settings",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "MODEL_NAME",
    "LLM_TEMPERATURE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "SECTION_MARKER",
    "VECTOR_K",
    "BM25_K",
    "FINAL_K",
    "RRF_C",
    "BM25_WEIGHT",
    "ENABLE_HYBRID",
    "ENABLE_MULTI_QUERY",
    "EMBEDDING_MODEL",
    "PERSIST_DIR",
    "COLLECTION_NAME",
    "FEISHU_WEBHOOK_URL",
]
