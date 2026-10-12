import re
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader


def clean_broken_text(text: str) -> str:
    """
    修复“一个字一行”的排版问题。
    如果一行只有一个字符（且不是标点），就把它和下一行拼接起来。
    """
    lines = text.split('\n')
    merged_lines = []
    temp_line = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 如果当前行长度 <= 1，且不是常见的段落标记，就暂存起来等待拼接
        if len(line) <= 1 and not line in ['。', '，', '！', '？', '；', '：']:
            temp_line += line
        else:
            # 遇到正常长度的行，把之前暂存的单字拼接到前面
            if temp_line:
                merged_lines.append(temp_line + line)
                temp_line = ""
            else:
                merged_lines.append(line)

    if temp_line:
        merged_lines.append(temp_line)

    # 把重新拼好的行用换行符连接
    return "\n".join(merged_lines)


def load_documents(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        # 对每一页的内容进行“拼字”修复
        for doc in docs:
            doc.page_content = clean_broken_text(doc.page_content)
        return docs
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    else:
        raise ValueError("仅支持 PDF 和 TXT 文件")