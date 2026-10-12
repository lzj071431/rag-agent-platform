import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, SECTION_MARKER

# 合并文件里的分节标题：===== 文件：<文档全名> =====
SECTION_RE = re.compile(
    rf"^{re.escape(SECTION_MARKER)}\s*文件：(?P<title>.*?)\s*{re.escape(SECTION_MARKER)}\s*$",
    re.M,
)


def _split_sections(text: str):
    """把合并文本切成 [(文档标题, 正文), ...]。

    标题来自 merge_txt.py 写入的 "===== 文件：xxx =====" 分节标记；
    如果文本里没有标记（例如用户直接上传单份 PDF/TXT），
    则整体作为一节，标题为 None。
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group("title").strip(), text[start:end]))
    return sections


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks = []
    for doc in documents:
        src_name = doc.metadata.get("source") or doc.metadata.get("file_name") or "未知来源"
        for title, body in _split_sections(doc.page_content):
            if not body.strip():
                continue
            sub = Document(page_content=body, metadata=dict(doc.metadata))
            for c in splitter.split_documents([sub]):
                # === 核心修复：给每个 chunk 补上"它属于哪份文档"的上下文 ===
                # 财报正文里常常不重复出现公司名（例如"行权价格为每份人民币15.83元"
                # 所在段落通篇没有"妙可蓝多"三个字），导致问"妙可蓝多的行权价格"时
                # 向量检索无法把该段落与问题关联上。这里统一加上前缀。
                if title and not c.page_content.startswith("【来源】"):
                    c.page_content = f"【来源】{title}\n{c.page_content}"
                c.metadata["source"] = src_name
                if title:
                    c.metadata["section"] = title
                chunks.append(c)

    # 给每个 chunk 分配唯一 ID
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    lens = [len(c.page_content) for c in chunks] or [0]
    print("=" * 66)
    print(f"[切分] chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print(f"[切分] 输入文档 {len(documents)} 篇 -> 输出文本块 {len(chunks)} 个")
    print(f"[切分] 块长度 min={min(lens)} / avg={sum(lens) // len(lens)} / max={max(lens)}")
    print("=" * 66)
    return chunks
