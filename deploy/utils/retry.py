"""Tenacity 重试策略统一封装。

对应需求第 3 项：大模型 / 检索 / Agent 调用的指数退避重试。

策略
----
* 最多重试 ``settings.retry_max_attempts`` 次（默认 3）；
* 指数退避：2s → 4s → 8s（``multiplier=2, exp_base=2``），并叠加随机抖动
  避免多实例同时重试造成"惊群"；
* 单次等待上限 ``settings.retry_max_wait``（默认 30s）；
* 只对**可恢复错误**重试：网络超时、连接错误、5xx、429。
  4xx（如 401 鉴权失败、400 参数错误）属于调用方问题，重试没有意义，直接失败。

为什么不用 SDK 自带的 ``max_retries``？
--------------------------------------
``ChatOpenAI`` 的 ``max_retries`` 只覆盖 HTTP 层，且不打印结构化日志；
而检索（Chroma）、CrewAI 任务等根本不是 HTTP 调用。统一用 Tenacity
可以在**同一处**定义策略、统一记录每次重试的日志与耗时，便于线上排障。
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

#: 明确可重试的异常类型（网络层 + OpenAI 兼容层）。
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    ConnectionError,
    TimeoutError,
)

#: 可重试的 HTTP 状态码：5xx 服务端错误 + 429 限流。
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _import_openai_errors() -> tuple[type[BaseException], ...]:
    """延迟导入 openai 异常类型，避免未安装时影响本模块导入。"""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai 是必装依赖
        return ()

    names = (
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
    )
    found: list[type[BaseException]] = []
    for name in names:
        exc = getattr(openai, name, None)
        if isinstance(exc, type) and issubclass(exc, BaseException):
            found.append(exc)
    return tuple(found)


_OPENAI_RETRYABLE = _import_openai_errors()


def _status_code_of(exc: BaseException) -> int | None:
    """尽力从各类异常中提取 HTTP 状态码。"""
    # httpx.HTTPStatusError / httpx.Response
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status

    # openai.APIStatusError 的 status_code 属性
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    return None


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    * 网络类异常 → 重试
    * HTTP 5xx / 429 → 重试
    * 其它 4xx（鉴权、参数错误）→ 不重试，重试只会浪费配额与时间
    """
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if _OPENAI_RETRYABLE and isinstance(exc, _OPENAI_RETRYABLE):
        return True

    status = _status_code_of(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES

    return False


def _before_sleep_log(retry_state: RetryCallState) -> None:
    """每次重试前记录结构化日志（含退避秒数与异常摘要）。"""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    next_wait = retry_state.next_action.sleep if retry_state.next_action else 0.0
    logger.warning(
        "调用失败，%.1fs 后重试 | 第 %d/%d 次 | 异常=%s: %s",
        next_wait,
        retry_state.attempt_number,
        settings.retry_max_attempts,
        type(exc).__name__ if exc else "Unknown",
        str(exc)[:200] if exc else "",
        extra={
            "event": "retry",
            "attempt": retry_state.attempt_number,
            "wait_seconds": round(next_wait, 2),
            "error_type": type(exc).__name__ if exc else "Unknown",
        },
    )


def build_retrying(
    *,
    max_attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
    exponential_base: float | None = None,
) -> Retrying:
    """按全局配置构造 :class:`tenacity.Retrying` 实例。

    显式传入参数时覆盖配置，便于特殊场景（如离线批处理放宽重试）。
    """
    attempts = max_attempts if max_attempts is not None else settings.retry_max_attempts
    first = initial_wait if initial_wait is not None else settings.retry_initial_wait
    cap = max_wait if max_wait is not None else settings.retry_max_wait
    base = (
        exponential_base
        if exponential_base is not None
        else settings.retry_exponential_base
    )

    return Retrying(
        stop=stop_after_attempt(attempts),
        # multiplier=first 保证首次等待约等于 first（2s），之后 4s、8s...
        # jitter 取 [0,1) 秒，打散并发重试的尖峰。
        wait=wait_exponential_jitter(
            initial=first, max=cap, exp_base=base, jitter=1
        ),
        retry=retry_if_exception(is_retryable),
        before_sleep=_before_sleep_log,
        reraise=True,
    )


def retryable(
    *,
    max_attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
    exponential_base: float | None = None,
    on_error: Callable[[BaseException], Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """装饰器：为函数加上指数退避重试。

    用法::

        @retryable()
        def call_llm(prompt: str) -> str:
            ...

    Args:
        max_attempts: 最多尝试次数（含首次）。
        initial_wait: 首次退避秒数。
        max_wait: 单次退避上限。
        exponential_base: 指数底数。
        on_error: 重试耗尽后的异常转换钩子；返回非 None 则作为结果返回。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retrying = build_retrying(
                max_attempts=max_attempts,
                initial_wait=initial_wait,
                max_wait=max_wait,
                exponential_base=exponential_base,
            )
            # 记录真实尝试次数：reraise=True 时 tenacity 不会抛 RetryError，
            # 无法从异常反推次数，因此在 before 钩子里逐次累加。
            attempt_counter = {"n": 0}
            original_before = retrying.before

            def _count_attempt(_retry_state) -> None:
                attempt_counter["n"] += 1
                if original_before is not None:
                    original_before(_retry_state)

            retrying.before = _count_attempt

            started = time.perf_counter()
            try:
                result = retrying(func, *args, **kwargs)
            except BaseException as exc:
                elapsed = time.perf_counter() - started
                if on_error is not None:
                    converted = on_error(exc)
                    if converted is not None:
                        return converted  # type: ignore[return-value]

                attempts = attempt_counter["n"] or 1
                retryable = is_retryable(exc)
                # 只有「可重试异常且确实重试过」才叫"重试耗尽"；
                # 不可重试的异常一次即失败，不应误报为尝试了 N 次。
                if retryable and attempts > 1:
                    logger.error(
                        "重试耗尽，最终失败 | 函数=%s | 尝试 %d 次 | 耗时 %.2fs | %s: %s",
                        func.__qualname__,
                        attempts,
                        elapsed,
                        type(exc).__name__,
                        str(exc)[:300],
                        extra={
                            "event": "retry_exhausted",
                            "function": func.__qualname__,
                            "attempts": attempts,
                            "elapsed_seconds": round(elapsed, 3),
                            "error_type": type(exc).__name__,
                        },
                    )
                else:
                    logger.error(
                        "调用失败（不可重试或首次即失败）| 函数=%s | 尝试 %d 次 | 耗时 %.2fs | %s: %s",
                        func.__qualname__,
                        attempts,
                        elapsed,
                        type(exc).__name__,
                        str(exc)[:300],
                        extra={
                            "event": "call_failed",
                            "function": func.__qualname__,
                            "attempts": attempts,
                            "retryable": retryable,
                            "elapsed_seconds": round(elapsed, 3),
                            "error_type": type(exc).__name__,
                        },
                    )
                raise
            else:
                elapsed = time.perf_counter() - started
                logger.debug(
                    "调用成功 | 函数=%s | 耗时 %.2fs", func.__qualname__, elapsed
                )
                return result

        return wrapper

    return decorator


def attempts_used(exc: BaseException) -> int:
    """从异常链中还原实际尝试次数（用于日志）。

    注意：``build_retrying`` 使用 ``reraise=True``，tenacity 最终抛出的是
    **原始异常**而不是 ``RetryError``，因此无法从异常对象反推次数。
    该函数仅作为兼容保留；需要准确次数请使用 :func:`retryable` 内部计数。
    """
    try:
        from tenacity import RetryError

        if isinstance(exc, RetryError):
            return exc.last_attempt.attempt_number
    except Exception:  # pragma: no cover
        pass
    return 1


def call_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
    **kwargs: Any,
) -> T:
    """函数式调用：对任意可调用对象套用重试策略。

    适用于无法加装饰器的场景（如 LangChain Runnable、lambda）。

    Example::

        answer = call_with_retry(chain.invoke, question)
    """
    retrying = build_retrying(
        max_attempts=max_attempts,
        initial_wait=initial_wait,
        max_wait=max_wait,
    )
    return retrying(func, *args, **kwargs)


__all__ = [
    "retryable",
    "call_with_retry",
    "build_retrying",
    "is_retryable",
]
