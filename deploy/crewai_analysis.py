"""向后兼容层 —— 旧版 ``crewai_analysis`` 模块。

实际实现已迁移到 :mod:`services.agent_service`。

保留本文件是为了让 ``from crewai_analysis import run_competitor_analysis``
等旧调用方式继续可用。
"""

from __future__ import annotations

from services.agent_service import (
    CompetitorAnalysisService,
    CompetitorReport,
    reset_vectorstore_cache,
    run_competitor_analysis,
    search_knowledge_base_impl,
)

#: 旧代码可能引用这些常量
from core.config import settings

RELEVANCE_DISTANCE_THRESHOLD = settings.retrieval.relevance_distance_threshold
REPORT_PATH = settings.report_output_path

__all__ = [
    "CompetitorAnalysisService",
    "CompetitorReport",
    "run_competitor_analysis",
    "reset_vectorstore_cache",
    "search_knowledge_base_impl",
    "RELEVANCE_DISTANCE_THRESHOLD",
    "REPORT_PATH",
]
