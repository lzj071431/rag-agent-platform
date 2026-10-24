"""文本清洗工具。

从原 ``loader.py`` 中抽出的纯函数，与"加载"动作解耦，便于单测与复用。
"""

from __future__ import annotations

import re

#: 中文行尾标点：这些字符单独成行时不应被拼接到下一行（它们是完整语义单元）。
_STANDALONE_PUNCTUATION = frozenset("。，！？；：、）】》”’…—")

#: 连续空白（保留换行）。
_MULTI_SPACE_RE = re.compile(r"[ \t\u3000]{2,}")


def clean_broken_text(text: str) -> str:
    """修复"一个字一行"的 PDF 排版问题。

    如果一行只有一个字符（且不是标点），就把它和下一行拼接起来。

    Args:
        text: PDF 页面抽取出的原始文本。

    Returns:
        修复后的文本（行已合并，空行已移除）。
    """
    if not text:
        return ""

    merged_lines: list[str] = []
    pending = ""

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # 单字符且非标点 → 暂存，等待与下一行拼接
        if len(line) <= 1 and line not in _STANDALONE_PUNCTUATION:
            pending += line
        else:
            if pending:
                merged_lines.append(pending + line)
                pending = ""
            else:
                merged_lines.append(line)

    if pending:
        merged_lines.append(pending)

    return "\n".join(merged_lines)


def normalize_whitespace(text: str) -> str:
    """压缩行内多余空格/全角空格，保留换行结构。"""
    if not text:
        return ""
    return _MULTI_SPACE_RE.sub(" ", text)


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """安全截断（超长时追加省略号）。"""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


def ensure_prefix(content: str, prefix: str) -> str:
    """确保内容以指定前缀开头（已存在则不重复添加）。"""
    if not prefix:
        return content
    if content.startswith(prefix):
        return content
    return f"{prefix}{content}"


__all__ = [
    "clean_broken_text",
    "normalize_whitespace",
    "truncate",
    "ensure_prefix",
]
