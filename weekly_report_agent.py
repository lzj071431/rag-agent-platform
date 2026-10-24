"""向后兼容层 —— 旧版 ``weekly_report_agent`` 模块。

实际实现已迁移到 :mod:`services.report_service`。

本模块保留旧版**全部公开名字**，确保历史脚本 / 文档 / 面试演示代码
可以直接 ``from weekly_report_agent import app, fetch_tasks`` 而不失效：

* ``fetch_tasks`` / ``analyze_tasks`` / ``generate_report`` / ``send_report``
  —— 旧版 LangGraph 节点函数（现在是绑定了默认服务的可调用对象）
* ``_build_card`` —— 旧版飞书卡片构建
* ``workflow`` / ``app`` —— 旧版工作流对象与编译后的图
* ``FEISHU_WEBHOOK`` / ``MAX_CONTENT`` / ``llm`` / ``http_client`` —— 旧版模块级变量

命令行用法保持不变::

    python weekly_report_agent.py
"""

from __future__ import annotations

from typing import Any

from core.config import settings
from core.logger import get_logger, setup_logging
from services.report_service import (
    WeeklyReportResult,
    WeeklyReportService,
    WeeklyReportState,
    build_weekly_report_graph,
    default_task_provider,
)

setup_logging()
logger = get_logger(__name__)

#: 兼容旧变量：飞书 Webhook 与正文长度上限
FEISHU_WEBHOOK: str = settings.feishu_webhook_url.get_secret_value()
MAX_CONTENT: int = settings.webhook_max_content


# ======================================================================
# 节点函数（旧签名：接收并返回 state）
# ======================================================================
# 注意：不能直接暴露 WeeklyReportService._fetch_tasks —— 那是未绑定的
# 实例方法，调用 fetch_tasks(state) 会少传 self 而抛 TypeError。
# 这里用默认服务实例绑定后包装成普通函数。
_default_service = WeeklyReportService(send_to_feishu=True)


def fetch_tasks(state: WeeklyReportState) -> WeeklyReportState:
    """节点1：获取本周任务（兼容旧接口）。"""
    return _default_service._fetch_tasks(state)


def analyze_tasks(state: WeeklyReportState) -> WeeklyReportState:
    """节点2：分析任务完成情况（兼容旧接口）。"""
    return _default_service._analyze_tasks(state)


def generate_report(state: WeeklyReportState) -> WeeklyReportState:
    """节点3：调用 LLM 生成周报（兼容旧接口）。"""
    return _default_service._generate_report(state)


def send_report(state: WeeklyReportState) -> WeeklyReportState:
    """节点4：发送周报到飞书（兼容旧接口）。"""
    return _default_service._send_report(state)


def _build_card(report: str) -> dict[str, Any]:
    """构建飞书消息卡片（兼容旧接口）。"""
    from utils.webhook import build_feishu_card

    return build_feishu_card(report, max_content=MAX_CONTENT)


#: 兼容旧变量：模块级 LLM（惰性代理，避免导入即要求 API Key）
class _LazyLLM:
    """惰性 LLM 代理：首次访问时才创建客户端。

    旧版在导入时直接构造 ChatOpenAI，导致"没配 Key 连 import 都失败"。
    """

    _inner: Any = None

    def _get(self) -> Any:
        if self._inner is None:
            from services.llm_factory import create_chat_model

            self._inner = create_chat_model()
        return self._inner

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get(), item)

    def __repr__(self) -> str:  # pragma: no cover
        return "<LazyLLM (未初始化)>"


llm = _LazyLLM()
http_client = None  # 旧版共享客户端；现在由 llm_factory 统一管理

#: 兼容旧变量：编译后的 LangGraph 图
workflow = _default_service.graph
app = workflow


if __name__ == "__main__":
    logger.info("🚀 启动自动周报生成 Agent...")
    result = WeeklyReportService(send_to_feishu=True).run()
    if result.status == "failed":
        logger.error("流程中存在失败节点: %s", result.error)
    else:
        logger.info("最终状态: %s | 耗时 %.2fs", result.status, result.elapsed_seconds)
        if result.report:
            logger.info("周报正文：\n%s", result.report)


__all__ = [
    "WeeklyReportService",
    "WeeklyReportResult",
    "WeeklyReportState",
    "build_weekly_report_graph",
    "default_task_provider",
    "fetch_tasks",
    "analyze_tasks",
    "generate_report",
    "send_report",
    "_build_card",
    "workflow",
    "app",
    "llm",
    "http_client",
    "FEISHU_WEBHOOK",
    "MAX_CONTENT",
]
