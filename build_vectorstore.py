from loader import load_documents
from splitter import split_documents
from vector_store import build_vectorstore

PDF_PATH = "finance_clean.pdf"
docs = load_documents(PDF_PATH)
chunks = split_documents(docs)
build_vectorstore(chunks)
print(f"向量库构建完成，共 {len(chunks)} 个 chunk")