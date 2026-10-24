"""全局结构化日志。

替换项目中所有 ``print()``。

设计要点
--------
1. **一次配置，处处生效**：:func:`setup_logging` 在进程入口调用一次，
   之后所有模块通过 :func:`get_logger` 取名即可，无需重复 ``basicConfig``。
2. **两条输出通道**：
   * 控制台 —— 本地开发阅读，``text`` 格式，带颜色级别；
   * 文件 —— 线上排障，``RotatingFileHandler`` 自动轮转，避免撑爆磁盘。
3. **上下文注入**：``session_id`` / ``request_id`` 这类追踪字段通过
   ``ContextVar`` 自动附加到每条日志，无需逐处传参。
4. **第三方库降噪**：httpx / urllib3 / chromadb 等的 INFO 日志过于嘈杂，
   统一压到 WARNING。
5. **可切换 JSON**：``LOG_FORMAT=json`` 时输出单行 JSON，便于 ELK / Loki 采集。

日志格式示例::

    2026-10-23 22:15:03,421 | INFO     | services.rag_service | [req=7f3a] 检索完成 | 命中 10 块 | 耗时 0.42s
"""

from __future__ import annotations

from core import bootstrap as _bootstrap  # noqa: F401  (确保离线开关先生效)

import json
import logging
import logging.handlers
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, MutableMapping

from core.config import settings

#: 追踪上下文：在 UI / API 入口设置一次，之后所有日志自动携带。
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="-")

#: 内部标记，确保 :func:`setup_logging` 幂等（Streamlit 会反复重跑脚本）。
_CONFIGURED = False

#: 需要降噪的第三方 logger。
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "chromadb",
    "chromadb.telemetry",
    "openai",
    "langchain",
    "langsmith",
    "crewai",
    "LiteLLM",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "posthog",
    "opentelemetry",
    "matplotlib",
    "PIL",
)

_RESET = "\033[0m"
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",  # bold red
}


class ContextFilter(logging.Filter):
    """把 ContextVar 中的追踪字段注入到每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        record.session_id = _session_id_var.get()
        return True


class TextFormatter(logging.Formatter):
    """控制台友好格式（带级别颜色）。"""

    def __init__(self, *, colorize: bool = True) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "[req=%(request_id)s] %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        if self.colorize and record.levelname in _LEVEL_COLORS:
            # 只给级别着色，避免整行花花绿绿影响阅读
            original = record.levelname
            record.levelname = f"{_LEVEL_COLORS[original]}{original}{_RESET}"
            try:
                return super().format(record)
            finally:
                record.levelname = original
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """单行 JSON 格式，便于日志采集系统解析。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # extra={...} 传入的自定义字段一并输出
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            if key not in payload:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


_RESERVED_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "request_id", "session_id"}


def setup_logging(
    level: str | int | None = None,
    *,
    force: bool = False,
) -> None:
    """初始化全局日志（幂等）。

    Args:
        level: 日志级别；默认取 ``settings.log_level``。
        force: 强制重新配置（测试用）。
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_level = level if level is not None else settings.log_level
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

    root = logging.getLogger()
    # 清掉既有 handler，避免重复输出（Streamlit 重跑时尤其明显）
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(resolved_level)

    context_filter = ContextFilter()
    use_json = settings.log_format == "json"

    # ---------- 控制台 ----------
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(resolved_level)
    console.setFormatter(
        JsonFormatter() if use_json else TextFormatter(colorize=sys.stdout.isatty())
    )
    console.addFilter(context_filter)
    root.addHandler(console)

    # ---------- 文件（轮转）----------
    if settings.log_file_enabled:
        try:
            log_dir = Path(settings.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / "app.log",
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(resolved_level)
            # 文件始终用 text 格式（带完整时间戳与模块信息），便于 grep
            file_handler.setFormatter(TextFormatter(colorize=False))
            file_handler.addFilter(context_filter)
            root.addHandler(file_handler)
        except OSError as exc:  # 只读文件系统等场景：降级为仅控制台
            root.warning("日志文件初始化失败，已降级为仅控制台输出: %s", exc)

    # ---------- 第三方库降噪 ----------
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug(
        "日志系统已初始化: level=%s format=%s file=%s",
        logging.getLevelName(resolved_level),
        settings.log_format,
        settings.log_file_enabled,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 logger。

    ⚠️ 本函数**不会**主动接管日志配置（不调用 :func:`setup_logging`）。

    原因：``utils.retry`` 等库模块在导入时就会调用 ``get_logger``。如果这里
    自动配置日志，那么"import 一个库"就会清空宿主的 root handler、改写日志级别
    —— 实测会破坏 pytest 的 ``caplog``、Jupyter 以及任何嵌入式宿主的日志设置。
    库应当只提供 logger，配置由应用入口显式完成。

    应用入口（``app.py`` / CLI 脚本）请先调用 :func:`setup_logging`。
    为方便独立脚本直接使用，若检测到 root **完全没有 handler**（说明从未初始化），
    则做一次兜底的默认配置。

    Args:
        name: 通常传 ``__name__``；省略则返回 root logger。

    Returns:
        标准 ``logging.Logger`` 实例。
    """
    root = logging.getLogger()
    if not root.handlers and not _CONFIGURED:
        # 宿主尚未配置任何日志：补一次默认配置，保证日志不丢失
        setup_logging()
    return logging.getLogger(name)


def set_request_context(request_id: str | None = None, session_id: str | None = None) -> None:
    """设置当前执行上下文的追踪 ID（自动附加到后续所有日志）。"""
    if request_id is not None:
        _request_id_var.set(request_id)
    if session_id is not None:
        _session_id_var.set(session_id)


def new_request_id() -> str:
    """生成一个短追踪 ID 并设为当前上下文。"""
    import uuid

    rid = uuid.uuid4().hex[:8]
    _request_id_var.set(rid)
    return rid


def log_timing(
    logger: logging.Logger,
    action: str,
    elapsed: float,
    *extra: Any,
    level: int = logging.INFO,
) -> None:
    """统一的耗时日志，便于线上做性能分析。

    示例输出::

        2026-10-23 22:15:03,421 | INFO | services.rag_service | 检索完成 | 耗时 0.42s | ...
    """
    parts = [f"{action}", f"耗时 {elapsed:.2f}s"]
    parts.extend(str(item) for item in extra)
    logger.log(level, " | ".join(parts))


class LoggerAdapter(logging.LoggerAdapter):
    """支持 ``logger.info("msg", extra={...})`` 风格的结构化字段。

    用法::

        log = get_adapter(__name__, stage="retrieval")
        log.info("检索完成", extra={"hits": 10})
    """

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


def get_adapter(name: str, **context: Any) -> LoggerAdapter:
    """获取带固定上下文字段的 logger 适配器。"""
    return LoggerAdapter(get_logger(name), context)


__all__ = [
    "setup_logging",
    "get_logger",
    "get_adapter",
    "set_request_context",
    "new_request_id",
    "log_timing",
    "ContextFilter",
    "TextFormatter",
    "JsonFormatter",
]
