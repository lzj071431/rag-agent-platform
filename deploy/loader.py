"""向后兼容层 —— 旧版 ``loader`` 模块。

实际实现已迁移到 :mod:`services.document_service`。
"""

from __future__ import annotations

from services.document_service import load_documents
from utils.text_cleaner import clean_broken_text

__all__ = ["load_documents", "clean_broken_text"]
