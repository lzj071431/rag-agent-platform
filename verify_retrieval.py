"""端到端验证：完全按 app.py 的流程走一遍（加载 -> 切分 -> 建库 -> 检索）"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from loader import load_documents
from splitter import split_documents
from vector_store import build_vectorstore, load_vectorstore
from rag_chain import build_rag_chain
from config import PERSIST_DIR

Q = "妙可蓝多的行权价格是多少？"
GOLD = "15.83"

docs = load_documents("competitor_data.txt")
for d in docs:
    d.metadata["source"] = "competitor_data.txt"
chunks = split_documents(docs)
print(f">>> 切分后 chunk 总数 = {len(chunks)}\n")

vs = build_vectorstore(chunks, PERSIST_DIR, reset=True)
print(f">>> 库内条数 = {vs._collection.count()}（应等于 chunk 总数 {len(chunks)}，不再有重复）\n")

vs = load_vectorstore(PERSIST_DIR)
chain, retriever = build_rag_chain(vs)

print("\n" + "=" * 70)
print(f"提问：{Q}")
print("=" * 70)
res = retriever.invoke(Q)
print(f"最终交给大模型的块数: {len(res)}")
hit_rank = None
for i, d in enumerate(res, 1):
    if GOLD in d.page_content:
        hit_rank = i
    mark = "   <<< 含 15.83（答案）" if GOLD in d.page_content else ""
    print(f"  #{i} | {d.page_content[:72].replace(chr(10), ' ')}{mark}")
print(f"\n>>> 答案块是否命中前 {len(res)}: {'是，第 %d 位' % hit_rank if hit_rank else '否'}")

# 试着真调一次 DeepSeek 拿最终回答（失败不影响检索结论）
print("\n" + "=" * 70)
print("调用 DeepSeek 生成最终回答：")
print("=" * 70)
try:
    ans = chain.invoke(Q)
    print(ans[:800])
except Exception as e:
    print(f"[跳过] LLM 调用失败（不影响检索验证）: {type(e).__name__}: {str(e)[:200]}")
