"""混合检索服务：向量召回 + BM25 召回 + RRF 融合。

从原 ``rag_chain.py`` 中抽出的检索逻辑，与"生成"解耦。

为什么用 RRF（Reciprocal Rank Fusion）？
---------------------------------------
向量相似度与 BM25 分数量纲完全不同，直接加权求和需要归一化，且对
分数分布敏感。RRF 只用**排名**：``score = Σ weight / (c + rank)``，
无需归一化，稳定可靠，是业界常用做法（``c`` 惯例取 60）。
"""

from __future__ import annotations

import time
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from core.config import settings
from core.exceptions import RetrievalError
from core.logger import get_logger
from utils.bm25 import SimpleBM25
from utils.retry import call_with_retry

logger = get_logger(__name__)


def dedupe_documents(docs: List[Document]) -> List[Document]:
    """按内容去重，保持原有顺序。"""
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in docs:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        unique.append(doc)
    return unique


def format_docs(docs: List[Document]) -> str:
    """把检索结果拼成上下文文本（去重后拼接）。

    去重很重要：重建向量库被重复写入时，避免同一段占满上下文窗口。
    """
    return "\n\n".join(doc.page_content for doc in dedupe_documents(docs))


class HybridRRFRetriever(BaseRetriever):
    """向量 Top-K 与 BM25 Top-K 用 RRF 融合的混合检索器。"""

    vectorstore: Any = Field(exclude=True)
    pages: List[Document] = Field(default_factory=list)
    bm25: Any = Field(default=None, exclude=True)

    k_vector: int = Field(default_factory=lambda: settings.retrieval.vector_k)
    k_bm25: int = Field(default_factory=lambda: settings.retrieval.bm25_k)
    k_final: int = Field(default_factory=lambda: settings.retrieval.final_k)
    rrf_c: int = Field(default_factory=lambda: settings.retrieval.rrf_c)
    bm25_weight: float = Field(
        default_factory=lambda: settings.retrieval.bm25_weight
    )
    use_hybrid: bool = Field(
        default_factory=lambda: settings.retrieval.enable_hybrid
    )

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """执行检索（LangChain 标准入口）。"""
        started = time.perf_counter()

        try:
            # ---------- 1. 向量召回 ----------
            vector_docs = dedupe_documents(
                call_with_retry(
                    self.vectorstore.similarity_search, query, k=self.k_vector
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("向量检索失败 | query=%s", query[:80])
            raise RetrievalError(
                "向量检索失败", detail=type(exc).__name__, original=exc
            ) from exc

        if not self.use_hybrid or self.bm25 is None:
            logger.debug(
                "纯向量检索 | 召回=%d | 返回=%d", len(vector_docs), self.k_final
            )
            return vector_docs[: self.k_final]

        # ---------- 2. BM25 召回 ----------
        try:
            scores = self.bm25.scores(query)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[
                : self.k_bm25
            ]
            bm25_docs = dedupe_documents(
                [self.pages[i] for i in order if scores[i] > 0]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BM25 检索失败，降级为纯向量检索 | %s: %s",
                type(exc).__name__,
                exc,
            )
            return vector_docs[: self.k_final]

        # ---------- 3. RRF 融合 ----------
        # 关键：每个排序列表都必须【先按内容去重再算名次】。
        # 否则同一个块在列表里出现两次时会各记一次分（1/(c+1) + 1/(c+2)），
        # 人为抬高分数，把仅靠另一路命中的正确答案挤出前 N。
        fused: dict[str, list] = {}
        for docs, weight in ((vector_docs, 1.0), (bm25_docs, self.bm25_weight)):
            for rank, doc in enumerate(docs, 1):
                key = doc.page_content
                if key not in fused:
                    fused[key] = [0.0, doc]
                fused[key][0] += weight / (self.rrf_c + rank)

        ranked = sorted(fused.values(), key=lambda item: -item[0])
        results = [doc for _, doc in ranked][: self.k_final]

        logger.info(
            "混合检索完成 | 向量召回=%d | BM25召回=%d | 融合后=%d | 耗时=%.2fs",
            len(vector_docs),
            len(bm25_docs),
            len(results),
            time.perf_counter() - started,
        )
        return results


def build_hybrid_retriever(vectorstore) -> HybridRRFRetriever:
    """从向量库中取出全部文本块，建立 BM25 索引，组装混合检索器。

    Args:
        vectorstore: 已加载的 ``Chroma`` 实例。

    Returns:
        可直接 ``invoke(query)`` 的检索器。

    Raises:
        RetrievalError: 向量库为空或读取失败。
    """
    cfg = settings.retrieval
    try:
        data = vectorstore.get(include=["documents", "metadatas"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取向量库内容失败")
        raise RetrievalError(
            "读取向量库内容失败", detail=type(exc).__name__, original=exc
        ) from exc

    texts = data.get("documents") or []
    metas = data.get("metadatas") or [{}] * len(texts)

    # 去重：历史遗留的重复块不应重复参与 BM25 打分
    seen: set[str] = set()
    pages: list[Document] = []
    for text, meta in zip(texts, metas):
        if not text or text in seen:
            continue
        seen.add(text)
        pages.append(Document(page_content=text, metadata=meta or {}))

    bm25 = SimpleBM25([p.page_content for p in pages]) if pages else None

    logger.info(
        "混合检索器就绪 | 库内块=%d | 去重后=%d | "
        "vector_k=%d bm25_k=%d final_k=%d bm25_weight=%.2f hybrid=%s",
        len(texts),
        len(pages),
        cfg.vector_k,
        cfg.bm25_k,
        cfg.final_k,
        cfg.bm25_weight,
        cfg.enable_hybrid,
    )
    if len(texts) != len(pages):
        logger.warning(
            "向量库存在 %d 条重复块！请在 Streamlit 上重新点击「构建向量库」（会先清空旧库）",
            len(texts) - len(pages),
        )

    if not pages:
        logger.warning("向量库为空，检索将返回空结果")

    return HybridRRFRetriever(vectorstore=vectorstore, pages=pages, bm25=bm25)


__all__ = [
    "HybridRRFRetriever",
    "build_hybrid_retriever",
    "dedupe_documents",
    "format_docs",
]
