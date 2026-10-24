"""Embedding 模型工厂。

Embedding 模型加载很慢（首次约 1~3 秒）且占用内存，因此全进程复用单例。

⚠️ 导入顺序约束（踩坑记录，勿改）
--------------------------------
本环境里 ``langchain_huggingface`` 会引用 ``langchain_core._api`` 中一个已被
移除的符号（``surface_langchain_deprecation_warnings``），**单独导入会直接
ImportError**；先导入 ``langchain_community`` 会完成该兼容性补丁。
因此下面必须先 import langchain_community。
"""

from __future__ import annotations

import threading

import langchain_community  # noqa: F401  (必须先于 langchain_huggingface 导入)
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings
from core.exceptions import VectorStoreError
from core.logger import get_logger

logger = get_logger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None
_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取全局 Embedding 实例（惰性加载 + 双检锁）。

    Returns:
        已配置为**仅本地加载**的 ``HuggingFaceEmbeddings``。

    Raises:
        VectorStoreError: 模型加载失败（通常是本地模型目录缺失）。
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    with _lock:
        if _embeddings is not None:
            return _embeddings
        try:
            logger.info(
                "正在加载 Embedding 模型 | model=%s | cache_folder=%s",
                settings.embedding_model,
                settings.models_dir,
            )
            _embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                cache_folder=settings.models_dir,
                # 彻底断网，只认本地模型目录
                model_kwargs={"local_files_only": True},
            )
            logger.info("Embedding 模型加载完成 | model=%s", settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - 统一转换为业务异常
            logger.exception("Embedding 模型加载失败 | model=%s", settings.embedding_model)
            raise VectorStoreError(
                "Embedding 模型加载失败",
                detail=f"model={settings.embedding_model}, dir={settings.models_dir}",
                original=exc,
            ) from exc

    return _embeddings


def reset_embeddings() -> None:
    """释放缓存的 Embedding 实例（测试或切换模型时使用）。"""
    global _embeddings
    with _lock:
        _embeddings = None
        logger.debug("已重置 Embedding 缓存")


__all__ = ["get_embeddings", "reset_embeddings"]
