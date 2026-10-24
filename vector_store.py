"""向后兼容层 —— 旧版 ``vector_store`` 模块。

实际实现已迁移到 :mod:`services.vector_store_service`。
"""

from __future__ import annotations

from services.embeddings import get_embeddings
from services.vector_store_service import (
    build_vectorstore,
    drop_collection,
    load_vectorstore,
    reset_vectorstore,
    vectorstore_exists,
)

__all__ = [
    "get_embeddings",
    "build_vectorstore",
    "load_vectorstore",
    "reset_vectorstore",
    "drop_collection",
    "vectorstore_exists",
]
