"""文档加载与切分服务。

合并原 ``loader.py`` 与 ``splitter.py`` 的职责，但保留两个独立的公开函数，
方便单独调用与测试。
"""

from __future__ import annotations

# ⚠️ 必须最先导入：在任何第三方库之前完成 HF 离线开关设置。
# langchain_community / huggingface_hub 在导入时会把 HF_HUB_OFFLINE 固化到
# huggingface_hub.constants（实测：先导入这些库再设环境变量，
# constants.HF_HUB_OFFLINE 仍是 False，离线约束失效）。
from core import bootstrap as _bootstrap  # noqa: F401  (导入即生效)

import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from core.exceptions import DocumentIngestionError
from core.logger import get_logger
from utils.text_cleaner import clean_broken_text

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt"})


def _build_section_re() -> re.Pattern[str]:
    """合并文件里的分节标题：``===== 文件：<文档全名> =====``。"""
    marker = re.escape(settings.chunking.section_marker)
    return re.compile(
        rf"^{marker}\s*文件：(?P<title>.*?)\s*{marker}\s*$",
        re.M,
    )


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """把合并文本切成 ``[(文档标题, 正文), ...]``。

    标题来自 ``merge_txt.py`` 写入的分节标记；如果文本里没有标记
    （例如用户直接上传单份 PDF/TXT），则整体作为一节，标题为 ``None``。
    """
    if not text:
        return []

    pattern = _build_section_re()
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group("title").strip(), text[start:end]))
    return sections


def load_documents(file_path: str | Path) -> list[Document]:
    """加载 PDF / TXT 文档。

    Args:
        file_path: 文件路径。

    Returns:
        LangChain ``Document`` 列表（PDF 按页返回）。

    Raises:
        DocumentIngestionError: 文件不存在、格式不支持或解析失败。
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentIngestionError("文件不存在", detail=str(path))

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentIngestionError(
            "仅支持 PDF 和 TXT 文件", detail=f"当前文件类型: {suffix or '未知'}"
        )

    logger.info("开始加载文档 | file=%s | type=%s", path.name, suffix)

    try:
        if suffix == ".pdf":
            docs = PyMuPDFLoader(str(path)).load()
            # 对每一页内容做"拼字"修复（修复"一个字一行"的 PDF 排版）
            for doc in docs:
                doc.page_content = clean_broken_text(doc.page_content)
        else:
            docs = TextLoader(str(path), encoding="utf-8").load()
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档加载失败 | file=%s", path.name)
        raise DocumentIngestionError(
            "文档加载失败", detail=path.name, original=exc
        ) from exc

    logger.info("文档加载完成 | file=%s | 页数/段数=%d", path.name, len(docs))
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """切分文档为文本块。

    关键处理（勿删）
    ----------------
    给每个 chunk 补上 ``【来源】<文件名>`` 前缀。财报正文里常常不重复出现公司名
    （例如"行权价格为每份人民币15.83元"所在段落通篇没有"妙可蓝多"三个字），
    导致问"妙可蓝多的行权价格"时向量检索无法把该段落与问题关联上。

    Args:
        documents: 原始文档列表。

    Returns:
        切分并补充元数据后的文本块列表。
    """
    if not documents:
        logger.warning("split_documents 收到空文档列表")
        return []

    cfg = settings.chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.size,
        chunk_overlap=cfg.overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        source_name = (
            doc.metadata.get("source")
            or doc.metadata.get("file_name")
            or "未知来源"
        )
        for title, body in split_sections(doc.page_content):
            if not body.strip():
                continue
            sub = Document(page_content=body, metadata=dict(doc.metadata))
            for chunk in splitter.split_documents([sub]):
                # 与原实现保持完全一致：只要内容已以「【来源】」开头就**不再**追加，
                # 而不是比较完整前缀是否相同。否则当同一段正文被不同标题分节时，
                # 会叠加成「【来源】B\n【来源】A\n…」，改变 embedding 输入。
                if title and not chunk.page_content.startswith("【来源】"):
                    chunk.page_content = f"【来源】{title}\n{chunk.page_content}"
                chunk.metadata["source"] = source_name
                if title:
                    chunk.metadata["section"] = title
                chunks.append(chunk)

    # 给每个 chunk 分配唯一 ID（评估脚本依赖 chunk_id 计算 Hit Rate / MRR）
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    lengths = [len(c.page_content) for c in chunks] or [0]
    logger.info(
        "文本切分完成 | 输入文档=%d 篇 | 输出文本块=%d 个 | "
        "chunk_size=%d | overlap=%d | 长度 min/avg/max = %d/%d/%d",
        len(documents),
        len(chunks),
        cfg.size,
        cfg.overlap,
        min(lengths),
        sum(lengths) // len(lengths),
        max(lengths),
    )
    return chunks


__all__ = ["load_documents", "split_documents", "split_sections", "SUPPORTED_SUFFIXES"]
