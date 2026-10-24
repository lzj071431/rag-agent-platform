"""CrewAI 多 Agent 竞品分析服务。

三个 Agent 依次协作：资深竞品调研员（调用 RAG 工具）→ 市场分析专家 → 专业商业报告撰写人。

保留的关键设计（勿删）
---------------------
1. **距离阈值防幻觉**：只有 L2 距离小于阈值（默认 1.05）的片段才被当作有效证据，
   否则返回"未找到"，从源头堵住"检索到一堆无关片段、再编造答案"的路径。
2. **向量库惰性加载 + 双检锁**：CrewAI 可能在一个 Agent 内并行发起多次工具调用，
   若两个线程同时初始化，Chroma 会因争抢同一 persist 目录报
   "Could not connect to tenant default_tenant"。
3. **每次新建 Agent/Task/Crew**：不复用模块级单例，避免上一轮执行状态污染下一轮。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import bootstrap as _bootstrap  # noqa: F401  (必须在 crewai 之前设置离线开关)
from core.config import settings
from core.exceptions import AgentExecutionError, ConfigurationError
from core.logger import get_logger
from core.prompts import (
    ANALYSIS_EXPECTED_OUTPUT,
    ANALYST_BACKSTORY,
    ANALYST_ROLE,
    REPORT_EXPECTED_OUTPUT,
    RESEARCHER_BACKSTORY,
    RESEARCHER_ROLE,
    WRITER_BACKSTORY,
    WRITER_ROLE,
    build_analysis_task_description,
    build_report_task_description,
    build_research_expected_output,
    build_research_task_description,
)
from services.llm_factory import create_crewai_llm
from services.vector_store_service import load_vectorstore
from utils.retry import call_with_retry

logger = get_logger(__name__)


@dataclass
class CompetitorReport:
    """竞品分析结果。"""

    content: str = ""
    product: str = ""
    competitor: str = ""
    elapsed_seconds: float = 0.0
    output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# 向量库缓存（线程安全）
# ======================================================================
_vectorstore = None
_vectorstore_lock = threading.Lock()


def _get_vectorstore():
    """惰性加载并缓存向量库，避免每次调用工具都重新加载 embedding 模型。"""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _vectorstore_lock:
        if _vectorstore is None:
            logger.info("CrewAI 工具：正在加载向量库 | dir=%s", settings.persist_dir)
            _vectorstore = load_vectorstore()
    return _vectorstore


def reset_vectorstore_cache() -> None:
    """丢弃已缓存的向量库连接。

    侧边栏重建知识库后调用，避免 CrewAI 工具继续查旧的库。
    """
    global _vectorstore
    with _vectorstore_lock:
        _vectorstore = None
    logger.info("CrewAI 工具：向量库缓存已重置")


# ======================================================================
# RAG 检索工具
# ======================================================================
def search_knowledge_base_impl(query: str, k: int = 5) -> str:
    """从本地知识库检索相关片段（工具的实际实现，便于单测）。

    Args:
        query: 完整的搜索查询语句。
        k: 召回条数。

    Returns:
        拼接后的文档片段；无有效命中时返回"知识库中未找到相关信息。"
    """
    threshold = settings.retrieval.relevance_distance_threshold
    vectorstore = _get_vectorstore()

    results = call_with_retry(
        vectorstore.similarity_search_with_score, query, k=k
    )
    docs = [doc for doc, score in results if score <= threshold]

    logger.debug(
        "知识库检索 | query=%s | 召回=%d | 阈值内=%d | 阈值=%.2f",
        query[:60],
        len(results),
        len(docs),
        threshold,
    )

    if not docs:
        return "知识库中未找到相关信息。"

    return "\n\n".join(
        f"[来源 {i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def _build_tool():
    """构建 CrewAI Tool（延迟导入 crewai，避免导入本模块即加载重依赖）。"""
    try:
        from crewai.tools import tool
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "未安装 crewai", detail="请执行 pip install -r requirements.txt"
        ) from exc

    @tool("search_knowledge_base")
    def search_knowledge_base(query: str) -> str:
        """从本地财务报告知识库中检索相关信息。输入一个完整的搜索查询语句，返回最相关的文档片段。

        Args:
            query: 完整的搜索查询语句，例如"妙可蓝多2025年股票期权的行权价格"。
        """
        return search_knowledge_base_impl(query)

    return search_knowledge_base


# ======================================================================
# 服务
# ======================================================================
class CompetitorAnalysisService:
    """竞品分析多 Agent 服务。"""

    def __init__(self, *, output_path: str | None = None) -> None:
        self._output_path = output_path or settings.report_output_path

    def _build_crew(self, product: str, competitor: str, verbose: bool):
        """每次调用都新建 Agent / Task / Crew。

        不复用模块级单例：Streamlit 每次交互都会重新执行脚本，复用同一个 Agent
        会把上一轮的执行状态带进来，导致跨次运行互相污染。
        """
        try:
            from crewai import Agent, Crew, Process, Task
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "未安装 crewai", detail="请执行 pip install -r requirements.txt"
            ) from exc

        llm = create_crewai_llm()
        search_tool = _build_tool()

        researcher = Agent(
            role=RESEARCHER_ROLE,
            goal=(
                f"基于本地知识库中的财务报告，全面搜集关于 {product} 及其竞品 "
                f"{competitor} 的功能、定价、用户评价等信息"
            ),
            backstory=RESEARCHER_BACKSTORY,
            llm=llm,
            tools=[search_tool],  # 只有调研员挂载 RAG 工具
            allow_delegation=False,
            verbose=verbose,
        )

        analyst = Agent(
            role=ANALYST_ROLE,
            goal=(
                f"基于调研员提供的资料，对 {product} 和 {competitor} "
                "进行深度对比分析，挖掘差异化机会"
            ),
            backstory=ANALYST_BACKSTORY,
            llm=llm,
            allow_delegation=False,
            verbose=verbose,
        )

        writer = Agent(
            role=WRITER_ROLE,
            goal="将分析结果整合成一份逻辑清晰、重点突出的专业竞品分析报告",
            backstory=WRITER_BACKSTORY,
            llm=llm,
            allow_delegation=False,
            verbose=verbose,
        )

        research_task = Task(
            description=build_research_task_description(product, competitor),
            expected_output=build_research_expected_output(product, competitor),
            agent=researcher,
        )

        analysis_task = Task(
            description=build_analysis_task_description(product, competitor),
            expected_output=ANALYSIS_EXPECTED_OUTPUT,
            agent=analyst,
            context=[research_task],
        )

        report_task = Task(
            description=build_report_task_description(product, competitor),
            expected_output=REPORT_EXPECTED_OUTPUT,
            agent=writer,
            context=[research_task, analysis_task],
            output_file=self._output_path,
        )

        return Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, report_task],
            process=Process.sequential,  # researcher -> analyst -> writer
            verbose=verbose,
        )

    def run(
        self,
        product: str,
        competitor: str,
        *,
        verbose: bool = False,
        max_attempts: int | None = None,
    ) -> CompetitorReport:
        """执行一轮竞品分析。

        Args:
            product: 分析的产品名，例如"妙可蓝多"。
            competitor: 竞品名，例如"伊利"。
            verbose: 是否输出 Agent 协作过程（命令行用 True，网页端用 False）。
            max_attempts: 覆盖重试次数。

        Returns:
            :class:`CompetitorReport`，报告同时落盘到 ``output_path``。

        Raises:
            ConfigurationError: 未配置 API Key。
            AgentExecutionError: 执行失败（重试耗尽）。
        """
        product = (product or "").strip()
        competitor = (competitor or "").strip()
        if not product or not competitor:
            raise AgentExecutionError(
                "参数不合法", detail="分析产品与竞品名称不能为空"
            )

        # 单线程预热，避免 Agent 并发调用工具时重复初始化向量库
        try:
            _get_vectorstore()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "向量库预热失败，Agent 将在调用工具时重试 | %s: %s",
                type(exc).__name__,
                exc,
            )

        started = time.perf_counter()
        logger.info(
            "竞品分析启动 | product=%s | competitor=%s | verbose=%s",
            product,
            competitor,
            verbose,
        )

        crew = self._build_crew(product, competitor, verbose)

        # ⚠️ 这里**不做整体重试**。
        # crew.kickoff 一次会串行跑 3 个 Agent、产生 3 组计费调用，且会写
        # output_file；整体重试等于「付 3 倍费用 + 重复写文件」，而且前两个
        # Agent 的产出在重试时会被全部丢弃重来（非幂等）。
        # 真正需要容错的是**单次 LLM 调用**——那已在 utils.retry 中处理
        # （CrewAI 的每次 LLM 请求都会走 SDK，且我们把 SDK 重试关为 0，
        # 由 llm_factory 保证超时，失败会快速暴露而不是静默重跑整条链）。
        # 因此若调用方显式要求重试，仅在此处按需开启，默认 1 次。
        try:
            result = call_with_retry(
                crew.kickoff,
                {"product": product, "competitor": competitor},
                max_attempts=max_attempts if max_attempts is not None else 1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("竞品分析执行失败 | product=%s", product)
            raise AgentExecutionError(
                "竞品分析执行失败",
                detail=f"{type(exc).__name__}: {str(exc)[:200]}",
                original=exc,
            ) from exc

        elapsed = time.perf_counter() - started
        content = getattr(result, "raw", None) or str(result)

        # 只认本轮真正写出的文件：先记录起始 mtime，避免把上一轮遗留的旧报告
        # 误报为本次产物。
        output_file = Path(self._output_path)
        output_exists = output_file.is_file() and output_file.stat().st_mtime >= started
        logger.info(
            "竞品分析完成 | 耗时=%.2fs | 报告长度=%d | 落盘=%s",
            elapsed,
            len(content),
            output_exists,
        )

        return CompetitorReport(
            content=content,
            product=product,
            competitor=competitor,
            elapsed_seconds=elapsed,
            output_path=str(output_file) if output_exists else None,
        )


def run_competitor_analysis(
    product: str,
    competitor: str,
    verbose: bool = True,
) -> str:
    """向后兼容旧接口：执行竞品分析并返回 Markdown 报告字符串。

    Args:
        product: 分析的产品名，例如"妙可蓝多"。
        competitor: 竞品名，例如"伊利"。
        verbose: 是否把 Agent 协作过程打印到终端。

    Returns:
        Markdown 报告字符串；报告同时写入 ``competitor_analysis_report.md``。
    """
    service = CompetitorAnalysisService()
    return service.run(product, competitor, verbose=verbose).content


__all__ = [
    "CompetitorAnalysisService",
    "CompetitorReport",
    "run_competitor_analysis",
    "reset_vectorstore_cache",
    "search_knowledge_base_impl",
]
