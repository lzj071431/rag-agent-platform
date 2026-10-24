"""统一业务异常体系。

设计要点：
* 所有异常继承 :class:`AppError`，调用方（UI / CLI / 未来的 HTTP API）只需
  捕获一个基类即可，不必感知底层是 httpx、openai 还是 chromadb 抛的错。
* 保留 ``original`` 字段挂载原始异常，排障时不丢失堆栈上下文。
* 异常消息面向"人"（可直接展示给用户），技术细节放 ``detail``。
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """业务异常基类。"""

    def __init__(
        self,
        message: str,
        *,
        detail: Any = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.original = original

    def __str__(self) -> str:  # pragma: no cover - 展示用
        if self.detail:
            return f"{self.message}（{self.detail}）"
        return self.message


class ConfigurationError(AppError):
    """配置缺失或非法（如未设置 API Key）。"""


class LLMInvocationError(AppError):
    """大模型调用失败（重试耗尽后抛出）。"""


class RetrievalError(AppError):
    """检索失败。"""


class DocumentIngestionError(AppError):
    """文档解析 / 切分失败。"""


class VectorStoreError(AppError):
    """向量库构建 / 加载 / 清理失败。"""


class WebhookError(AppError):
    """Webhook（飞书）发送失败。"""


class AgentExecutionError(AppError):
    """多 Agent 工作流执行失败。"""
