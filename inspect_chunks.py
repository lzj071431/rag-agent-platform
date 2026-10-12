# inspect_chunks.py
from loader import load_documents
from splitter import split_documents

# 确保这里写的是你清洗过的那个文件
PDF_PATH = "finance_clean.pdf"

docs = load_documents(PDF_PATH)
chunks = split_documents(docs)

print(f"共切分出 {len(chunks)} 个 chunk\n")

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{i}"
    print(f"--- chunk_{i} ---")
    print(chunk.page_content[:300])
    print()