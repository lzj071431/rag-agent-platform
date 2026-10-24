"""大模型客户端工厂。

统一创建 ``ChatOpenAI`` / CrewAI ``LLM`` 实例，并把**超时**与**重试**收口到这里。

设计要点
--------
1. **共享 httpx.Client**：避免每次调用新建连接池（原代码每次 ``httpx.Client()``
   都泄漏一个连接池，长期运行会耗尽文件句柄）。
2. **SDK 层重试关闭**：``max_retries=0``，由 ``utils.retry`` 统一控制退避节奏
   与日志，避免两层重试叠加导致"3×3=9 次"的意外放大。
3. **超时显式配置**：原代码未设超时，网络异常时会无限挂起。
"""

from __future__ import annotations

import threading
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from core.config import settings
from core.exceptions import ConfigurationError
from core.logger import get_logger

logger = get_logger(__name__)

_http_client: httpx.Client | None = None
_client_lock = threading.Lock()


def get_http_client() -> httpx.Client:
    """获取进程级共享的 httpx 客户端（线程安全单例）。"""
    global _http_client
    if _http_client is not None:
        return _http_client
    with _client_lock:
        if _http_client is None:
            _http_client = httpx.Client(
                timeout=httpx.Timeout(
                    settings.llm_timeout_seconds,
                    connect=min(10.0, settings.llm_timeout_seconds),
                ),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
            logger.debug("已创建共享 httpx.Client")
    return _http_client


def create_chat_model(
    *,
    temperature: float | None = None,
    streaming: bool = False,
    **overrides: Any,
) -> ChatOpenAI:
    """创建 LangChain 对话模型实例。

    Args:
        temperature: 采样温度；默认取配置。
        streaming: 是否流式输出。
        **overrides: 透传给 ``ChatOpenAI`` 的额外参数。

    Returns:
        可直接用于 LCEL 链的 ``ChatOpenAI``。

    Raises:
        ConfigurationError: 未配置 DEEPSEEK_API_KEY。
    """
    api_key = settings.require_llm_credentials()

    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "api_key": api_key,
        "base_url": settings.deepseek_base_url,
        "temperature": (
            temperature if temperature is not None else settings.llm_temperature
        ),
        "http_client": get_http_client(),
        # 关闭 SDK 内置重试：统一由 utils.retry 控制，避免重试次数相乘
        "max_retries": 0,
        "timeout": settings.llm_timeout_seconds,
        "streaming": streaming,
    }
    kwargs.update(overrides)

    logger.debug(
        "创建 ChatOpenAI | model=%s | base_url=%s | temperature=%s",
        kwargs["model"],
        kwargs["base_url"],
        kwargs["temperature"],
    )
    return ChatOpenAI(**kwargs)


def create_crewai_llm(*, temperature: float | None = None) -> Any:
    """创建 CrewAI 使用的 LLM（DeepSeek 的 OpenAI 兼容接口）。

    与 :func:`create_chat_model` 保持一致：显式设置 ``timeout`` 并把 SDK 层
    重试关到 0。CrewAI 底层默认 ``max_retries=2`` 且 ``timeout=None``，
    若不覆盖会与 ``utils.retry`` 的 3 次重试叠加成 6 次，且单次调用可能
    无限期挂起（实测默认值确为 ``max_retries=2, timeout=None``）。

    Returns:
        ``crewai.LLM`` 实例。

    Raises:
        ConfigurationError: 未配置 DEEPSEEK_API_KEY。
        ImportError: 未安装 crewai。
    """
    try:
        from crewai import LLM
    except ImportError as exc:  # pragma: no cover - crewai 是必装依赖
        raise ConfigurationError(
            "未安装 crewai", detail="请执行 pip install -r requirements.txt"
        ) from exc

    api_key = settings.require_llm_credentials()
    logger.debug("创建 CrewAI LLM | model=%s", settings.model_name)

    return LLM(
        model=settings.model_name,
        custom_openai=True,
        base_url=settings.deepseek_base_url,
        api_key=api_key,
        temperature=(
            temperature if temperature is not None else settings.llm_temperature
        ),
        # 与 create_chat_model 对齐：显式超时 + 关闭 SDK 重试
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def close_http_client() -> None:
    """关闭共享 HTTP 客户端（进程退出 / 测试清理时调用）。"""
    global _http_client
    with _client_lock:
        if _http_client is not None:
            _http_client.close()
            _http_client = None
            logger.debug("已关闭共享 httpx.Client")


__all__ = [
    "get_http_client",
    "create_chat_model",
    "create_crewai_llm",
    "close_http_client",
]
