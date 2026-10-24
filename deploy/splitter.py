"""向后兼容层 —— 旧版 ``splitter`` 模块。

实际实现已迁移到 :mod:`services.document_service`。
"""

from __future__ import annotations

from services.document_service import split_documents, split_sections

__all__ = ["split_documents", "split_sections"]
