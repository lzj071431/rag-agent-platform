"""Webhook 发送工具（飞书群机器人）。

从原 ``weekly_report_agent.py`` 的 ``_build_card`` / ``send_report`` 中抽出，
与 LangGraph 工作流解耦，便于在其它流程中复用。

关键细节（沿用原实现的踩坑经验）
--------------------------------
飞书机器人即使业务失败，HTTP 状态码通常也是 200，真正的结果在响应体
``code`` 字段里。因此**必须以 ``body["code"] == 0`` 判定成功**，
不能只看 ``resp.status_code``。
"""

from __future__ import annotations

from typing import Any

import httpx

from core.config import settings
from core.exceptions import WebhookError
from core.logger import get_logger
from utils.retry import call_with_retry

logger = get_logger(__name__)


def build_feishu_card(
    content: str,
    *,
    title: str = "📊 本周工作周报",
    template: str = "blue",
    max_content: int | None = None,
) -> dict[str, Any]:
    """构建飞书交互式消息卡片。

    Args:
        content: Markdown 正文。
        title: 卡片标题。
        template: 卡片主题色（blue / green / red / ...）。
        max_content: 正文字符上限，默认取配置。

    Returns:
        可直接作为 JSON body 发送的字典。
    """
    limit = max_content if max_content is not None else settings.webhook_max_content
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": template,
            },
            "elements": [{"tag": "markdown", "content": content[:limit]}],
        },
    }


def send_feishu_card(
    content: str,
    *,
    webhook_url: str | None = None,
    title: str = "📊 本周工作周报",
    timeout: float | None = None,
) -> dict[str, Any]:
    """发送卡片到飞书群（带重试）。

    Args:
        content: Markdown 正文。
        webhook_url: 覆盖默认 Webhook 地址（测试用）。
        title: 卡片标题。
        timeout: 请求超时秒数。

    Returns:
        飞书返回的响应体字典。

    Raises:
        ConfigurationError: 未配置 Webhook。
        WebhookError: 发送失败或飞书返回非 0 错误码。
    """
    url = (webhook_url or "").strip()
    if not url:
        # 未配置时 require_webhook 会抛出带修复指引的 ConfigurationError
        url = settings.require_webhook()

    payload = build_feishu_card(content, title=title)
    request_timeout = timeout if timeout is not None else settings.webhook_timeout_seconds
    http_timeout = httpx.Timeout(request_timeout, connect=min(5.0, request_timeout))

    def _post() -> dict[str, Any]:
        resp = httpx.post(url, json=payload, timeout=http_timeout)
        # 5xx / 429 由 retry 判定为可重试
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as exc:
            raise WebhookError(
                "飞书返回内容不是合法 JSON",
                detail=resp.text[:200],
                original=exc,
            ) from exc

    try:
        body = call_with_retry(_post)
    except WebhookError:
        raise
    except httpx.TimeoutException as exc:
        logger.exception("发送飞书超时 | timeout=%.1fs", request_timeout)
        raise WebhookError("发送飞书消息超时", detail=f"timeout={request_timeout}s", original=exc) from exc
    except httpx.HTTPError as exc:
        logger.exception("发送飞书失败 | %s", type(exc).__name__)
        raise WebhookError(
            "发送飞书消息失败", detail=type(exc).__name__, original=exc
        ) from exc

    code = body.get("code")
    if code == 0:
        logger.info("飞书消息发送成功 | 正文长度=%d", len(content))
        return body

    logger.error(
        "飞书返回业务错误 | http=200 code=%s msg=%s", code, body.get("msg")
    )
    raise WebhookError(
        "飞书返回业务错误",
        detail=f"code={code}, msg={body.get('msg')}",
    )


__all__ = ["build_feishu_card", "send_feishu_card"]
