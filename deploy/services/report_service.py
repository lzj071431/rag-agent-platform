"""周报生成服务（LangGraph 工作流）。

工作流：``fetch_tasks → analyze_tasks → generate_report → send_report``

与原 ``weekly_report_agent.py`` 的差异
-------------------------------------
1. **纯服务化**：原文件在模块导入时就构建工作流、初始化 LLM（导入即联网），
   现在改为惰性构建，导入本模块零副作用。
2. **数据获取可替换**：``fetch_tasks`` 抽成可注入的函数，接真实任务系统时
   只需传入自己的 provider，不用改图结构。
3. **错误不中断整条链**：发送失败通过 state 传递，而不是抛异常炸掉整个流程。
4. **结构化日志 + 重试**：LLM 调用与 Webhook 均带指数退避。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from core.exceptions import LLMInvocationError
from core.logger import get_logger
from core.prompts import WEEKLY_REPORT_TEMPLATE
from services.llm_factory import create_chat_model
from utils.retry import call_with_retry

logger = get_logger(__name__)


# ======================================================================
# 状态定义
# ======================================================================
class WeeklyReportState(TypedDict, total=False):
    """LangGraph 工作流状态。

    ``total=False`` 让各节点可以只返回自己更新的字段（LangGraph 会做增量合并）。
    ``error`` 必须声明，否则 LangGraph 更新状态时会报错。
    """

    tasks: List[dict]
    stats: dict
    report: str
    status: str
    error: str


@dataclass
class WeeklyReportResult:
    """周报生成结果。"""

    report: str = ""
    stats: dict = field(default_factory=dict)
    status: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    sent: bool = False


# ======================================================================
# 数据源
# ======================================================================
def default_task_provider() -> List[dict]:
    """默认任务数据源（模拟数据）。

    接入真实系统（Jira / 飞书多维表格）时，实现一个返回同样结构的函数，
    通过 :class:`WeeklyReportService` 构造参数注入即可。
    """
    return [
        {"id": 1, "title": "完成 RAG 系统评估", "status": "done"},
        {"id": 2, "title": "优化 Reranker 集成", "status": "done"},
        {"id": 3, "title": "编写 README 文档", "status": "done"},
        {"id": 4, "title": "调研 LangGraph 工作流", "status": "in_progress"},
        {"id": 5, "title": "准备技术分享 PPT", "status": "todo"},
    ]


# ======================================================================
# 服务
# ======================================================================
class WeeklyReportService:
    """封装 LangGraph 工作流的构建与执行。"""

    def __init__(
        self,
        task_provider: Callable[[], List[dict]] | None = None,
        *,
        send_to_feishu: bool = True,
    ) -> None:
        self._task_provider = task_provider or default_task_provider
        self._send_to_feishu = send_to_feishu
        self._app = None  # 惰性构建

    # ---------------- 节点实现 ----------------
    def _fetch_tasks(self, state: WeeklyReportState) -> WeeklyReportState:
        """节点1：获取本周任务。"""
        tasks = self._task_provider()
        logger.info("周报节点1 获取任务 | 共 %d 项", len(tasks))
        return {"tasks": tasks, "status": "tasks_fetched"}

    def _analyze_tasks(self, state: WeeklyReportState) -> WeeklyReportState:
        """节点2：统计分析完成情况。"""
        tasks = state.get("tasks") or []
        total = len(tasks)
        done = [t for t in tasks if t.get("status") == "done"]
        in_progress = [t for t in tasks if t.get("status") == "in_progress"]
        todo = [t for t in tasks if t.get("status") == "todo"]
        completion_rate = len(done) / total * 100 if total > 0 else 0.0

        stats = {
            "total": total,
            "done": len(done),
            "in_progress": len(in_progress),
            "todo": len(todo),
            "completion_rate": round(completion_rate, 1),
            "done_list": [t.get("title", "") for t in done],
            "in_progress_list": [t.get("title", "") for t in in_progress],
            "todo_list": [t.get("title", "") for t in todo],
        }
        logger.info(
            "周报节点2 统计分析完成 | 共 %d 项 | 完成 %d | 进行中 %d | 未开始 %d | 完成率 %.1f%%",
            total,
            len(done),
            len(in_progress),
            len(todo),
            completion_rate,
        )
        return {"stats": stats, "status": "analyzed"}

    def _generate_report(self, state: WeeklyReportState) -> WeeklyReportState:
        """节点3：调用大模型生成周报。"""
        stats = state.get("stats") or {}
        prompt = ChatPromptTemplate.from_template(WEEKLY_REPORT_TEMPLATE)

        try:
            chain = prompt | create_chat_model() | StrOutputParser()
            report = call_with_retry(
                chain.invoke,
                {
                    "total": stats.get("total", 0),
                    "done": stats.get("done", 0),
                    "in_progress": stats.get("in_progress", 0),
                    "todo": stats.get("todo", 0),
                    "completion_rate": stats.get("completion_rate", 0),
                    "done_list": "\n".join(
                        f"- {t}" for t in stats.get("done_list", [])
                    ),
                    "in_progress_list": "\n".join(
                        f"- {t}" for t in stats.get("in_progress_list", [])
                    ),
                    "todo_list": "\n".join(
                        f"- {t}" for t in stats.get("todo_list", [])
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("周报节点3 大模型调用失败")
            raise LLMInvocationError(
                "周报生成失败：大模型调用异常",
                detail=type(exc).__name__,
                original=exc,
            ) from exc

        logger.info("周报节点3 生成完成 | 正文长度=%d", len(report))
        return {"report": report, "status": "report_generated"}

    def _send_report(
        self, state: WeeklyReportState, *, enabled: bool | None = None
    ) -> WeeklyReportState:
        """节点4：发送周报到飞书。失败不抛异常，通过 status 向下游传递。

        Args:
            state: 工作流状态。
            enabled: 本次是否发送；``None`` 时取实例级开关。
        """
        should_send = self._send_to_feishu if enabled is None else enabled
        if not should_send:
            logger.info("周报节点4 已跳过发送（未启用）")
            return {"status": "skipped", "error": ""}

        report = (state.get("report") or "").strip()
        if not report:
            logger.warning("周报内容为空，跳过发送")
            return {"status": "skipped", "error": "empty report"}

        if not settings.has_webhook:
            logger.warning("未配置 FEISHU_WEBHOOK_URL，跳过发送")
            return {"status": "skipped", "error": "webhook not configured"}

        from utils.webhook import send_feishu_card

        try:
            send_feishu_card(report)
        except Exception as exc:  # noqa: BLE001
            logger.error("周报节点4 发送失败 | %s: %s", type(exc).__name__, exc)
            return {"status": "failed", "error": str(exc)}
        return {"status": "sent", "error": ""}

    # ---------------- 图构建 ----------------
    def _build_graph(self, *, send_to_feishu: bool | None = None):
        """构建并编译 LangGraph 工作流。

        Args:
            send_to_feishu: 本次运行的发送开关；``None`` 时取实例级配置。
        """
        from langgraph.graph import END, StateGraph

        def _send_node(state: WeeklyReportState) -> WeeklyReportState:
            return self._send_report(state, enabled=send_to_feishu)

        workflow = StateGraph(WeeklyReportState)
        workflow.add_node("fetch_tasks", self._fetch_tasks)
        workflow.add_node("analyze_tasks", self._analyze_tasks)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("send_report", _send_node)

        workflow.set_entry_point("fetch_tasks")
        workflow.add_edge("fetch_tasks", "analyze_tasks")
        workflow.add_edge("analyze_tasks", "generate_report")
        workflow.add_edge("generate_report", "send_report")
        workflow.add_edge("send_report", END)

        compiled = workflow.compile()
        logger.debug("周报 LangGraph 工作流构建完成")
        return compiled

    @property
    def graph(self):
        """暴露编译后的图（便于测试与可视化）。"""
        if self._app is None:
            self._app = self._build_graph()
        return self._app

    # ---------------- 执行 ----------------
    def run(self, *, send_to_feishu: bool | None = None) -> WeeklyReportResult:
        """执行完整周报流程。

        Args:
            send_to_feishu: 单次覆盖实例级发送开关（不修改实例状态）。

        Returns:
            :class:`WeeklyReportResult`。

        Raises:
            LLMInvocationError: 生成节点失败。
        """
        # 只在本次调用内生效，避免污染实例状态（同一实例多次 run 时行为不一致）
        send_flag = (
            send_to_feishu if send_to_feishu is not None else self._send_to_feishu
        )

        started = time.perf_counter()
        logger.info("周报工作流启动 | 发送飞书=%s", send_flag)

        initial: WeeklyReportState = {
            "tasks": [],
            "stats": {},
            "report": "",
            "status": "start",
            "error": "",
        }

        # 通过闭包把本次的发送开关传给节点，不改 self
        graph = self._build_graph(send_to_feishu=send_flag)
        final_state = graph.invoke(initial)
        elapsed = time.perf_counter() - started

        status = final_state.get("status", "")
        result = WeeklyReportResult(
            report=final_state.get("report", ""),
            stats=final_state.get("stats", {}),
            status=status,
            error=final_state.get("error", ""),
            elapsed_seconds=elapsed,
            sent=status == "sent",
        )
        logger.info(
            "周报工作流结束 | 状态=%s | 耗时=%.2fs | 正文长度=%d",
            status,
            elapsed,
            len(result.report),
        )
        return result


def build_weekly_report_graph(
    task_provider: Callable[[], List[dict]] | None = None,
    *,
    send_to_feishu: bool = True,
):
    """向后兼容：返回编译后的 LangGraph 图（旧 ``app`` 对象）。"""
    return WeeklyReportService(
        task_provider=task_provider, send_to_feishu=send_to_feishu
    ).graph


__all__ = [
    "WeeklyReportService",
    "WeeklyReportResult",
    "WeeklyReportState",
    "build_weekly_report_graph",
    "default_task_provider",
]
