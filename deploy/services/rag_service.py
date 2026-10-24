"""RAG 问答服务。

把"检索 → 组装上下文 → 调用大模型 → 解析答案"的完整编排收口在这里，
UI 只需调用 :class:`RAGService` 的少量方法。

与旧实现相比的改进
------------------
1. **返回结构化结果**：旧代码 UI 里只拿到一个字符串，无法展示引用来源与耗时；
   现在返回 :class:`RAGAnswer`，包含答案、来源文档、耗时、是否命中。
2. **空检索短路**：检索不到内容时直接返回兜底话术，**不浪费一次大模型调用**。
3. **重试**：LLM 调用与检索都套了 Tenacity 指数退避。
4. **可观测**：每个阶段都有耗时日志，便于线上定位慢在哪一步。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.config import settings
from core.exceptions import LLMInvocationError, RetrievalError
from core.logger import get_logger, log_timing
from core.prompts import RAG_NO_CONTEXT_ANSWER, RAG_QA_TEMPLATE
from services.llm_factory import create_chat_model
from services.retrieval_service import build_hybrid_retriever, format_docs
from utils.retry import call_with_retry

logger = get_logger(__name__)


@dataclass
class RAGAnswer:
    """一次问答的完整结果。"""

    answer: str
    sources: List[Document] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    retrieval_hit: bool = True

    def source_metadata(self, limit: int = 10) -> List[dict[str, Any]]:
        """整理来源元数据，便于 UI 渲染"引用来源"折叠面板。"""
        items: list[dict[str, Any]] = []
        for doc in self.sources[:limit]:
            items.append(
                {
                    "source": doc.metadata.get("source", "未知来源"),
                    "section": doc.metadata.get("section"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "preview": doc.page_content[:500],
                }
            )
        return items


class RAGService:
    """RAG 问答服务（面向一次会话的轻量封装）。

    用法::

        service = RAGService(vectorstore)
        result = service.ask("妙可蓝多的行权价格是多少？")
        print(result.answer, result.sources)
    """

    def __init__(self, vectorstore, *, build_retriever: bool = True) -> None:
        self._vectorstore = vectorstore
        self._retriever = build_hybrid_retriever(vectorstore) if build_retriever else None
        self._chain = self._build_chain() if build_retriever else None
        logger.debug("RAGService 初始化完成")

    # ------------------------------------------------------------------
    # 内部构建
    # ------------------------------------------------------------------
    def _build_chain(self):
        """构建 LCEL 链：{context, question} -> prompt -> llm -> str。"""
        prompt = ChatPromptTemplate.from_template(RAG_QA_TEMPLATE)
        llm = create_chat_model()

        retriever = self._retriever
        if settings.retrieval.enable_multi_query:
            retriever = self._wrap_multi_query(retriever, llm)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    @staticmethod
    def _wrap_multi_query(retriever, llm):
        """可选：MultiQueryRetriever（默认关闭）。

        让 LLM 把问题改写成多个 query 再分别检索，能提升召回，但每次提问
        会多花 1 次 API 调用、增加约 1~3 秒延迟。
        """
        # langchain 1.x 把 Retriever 系列挪到了 langchain_classic，这里做双路径兼容
        try:
            from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        except ModuleNotFoundError:  # pragma: no cover
            from langchain.retrievers.multi_query import MultiQueryRetriever

        logger.info("已启用 MultiQueryRetriever")
        return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    @property
    def retriever(self):
        """暴露检索器，便于评估脚本单独测检索指标。"""
        return self._retriever

    @property
    def chain(self):
        """暴露 LCEL 链，便于兼容旧调用方式。"""
        return self._chain

    def retrieve(self, question: str) -> List[Document]:
        """仅执行检索（不调用大模型）。

        Raises:
            RetrievalError: 检索失败。
        """
        if self._retriever is None:
            raise RetrievalError("检索器未初始化")
        started = time.perf_counter()
        docs = call_with_retry(self._retriever.invoke, question)
        log_timing(
            logger,
            "检索完成",
            time.perf_counter() - started,
            f"命中 {len(docs)} 块",
        )
        return docs

    def ask(self, question: str, *, top_k: int | None = None) -> RAGAnswer:
        """提问并返回结构化结果。

        Args:
            question: 用户问题。
            top_k: 仅返回前 N 条来源（不影响送进大模型的上下文数量）。

        Returns:
            :class:`RAGAnswer`。

        Raises:
            LLMInvocationError: 大模型调用在重试后仍失败。
            RetrievalError: 检索失败。
        """
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer="请输入问题。", retrieval_hit=False)

        started = time.perf_counter()

        # ---------- 1. 检索 ----------
        sources = self.retrieve(question)

        if not sources:
            logger.warning("检索结果为空，返回兜底回答 | question=%s", question[:80])
            return RAGAnswer(
                answer=RAG_NO_CONTEXT_ANSWER,
                sources=[],
                elapsed_seconds=time.perf_counter() - started,
                retrieval_hit=False,
            )

        # ---------- 2. 生成 ----------
        assert self._chain is not None
        try:
            answer = call_with_retry(self._chain.invoke, question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("大模型调用失败 | question=%s", question[:80])
            raise LLMInvocationError(
                "大模型调用失败，请稍后重试",
                detail=type(exc).__name__,
                original=exc,
            ) from exc

        elapsed = time.perf_counter() - started
        limit = top_k if top_k is not None else len(sources)
        log_timing(
            logger,
            "RAG 问答完成",
            elapsed,
            f"问题长度={len(question)}",
            f"上下文块数={len(sources)}",
            f"答案长度={len(answer)}",
        )
        return RAGAnswer(
            answer=answer,
            sources=sources[:limit],
            elapsed_seconds=elapsed,
            retrieval_hit=True,
        )


def build_rag_service(vectorstore) -> RAGService:
    """工厂函数：构建 RAG 服务（兼容旧 ``build_rag_chain`` 的调用习惯）。"""
    return RAGService(vectorstore)


def build_rag_chain(vectorstore):
    """向后兼容旧接口，返回 ``(chain, retriever)`` 元组。

    旧代码（评估脚本）依赖这种解包方式，保留以免破坏。
    """
    service = RAGService(vectorstore)
    return service.chain, service.retriever


__all__ = ["RAGService", "RAGAnswer", "build_rag_service", "build_rag_chain"]
