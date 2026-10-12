import math
import re
from collections import Counter
from typing import Any, List

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME,
    VECTOR_K, BM25_K, FINAL_K, RRF_C, BM25_WEIGHT, ENABLE_HYBRID, ENABLE_MULTI_QUERY,
)

load_dotenv()

PROMPT_TEMPLATE = """仅根据以下上下文回答问题。如果上下文中没有相关信息，请回答“根据提供的资料，我无法回答这个问题。”
上下文：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def format_docs(docs):
    # 去重后拼接：重建向量库被重复写入时，避免同一段占满上下文
    seen, uniq = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            uniq.append(d)
    return "\n\n".join(doc.page_content for doc in uniq)


# ======================================================================
# 关键词检索：字符二元组 BM25（纯 Python 实现，零额外依赖）
# 财报问答里大量是"15.83""行权价格"这类精确关键词，
# 纯向量检索对精确数字/专有名词不敏感，加一路 BM25 能显著提升命中。
# ======================================================================
def _grams(text: str) -> List[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]


class SimpleBM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_grams(t)) for t in corpus]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    def scores(self, query: str) -> List[float]:
        q = Counter(_grams(query))
        out = []
        for i, c in enumerate(self.docs):
            s = 0.0
            for t, qf in q.items():
                f = c.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * qf * (f * (self.k1 + 1)) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                    )
            out.append(s)
        return out


# ======================================================================
# 混合检索器：向量 Top-K 与 BM25 Top-K 用 RRF（倒数排名融合）合并
# RRF 只用"排名"不用"分数"，无需把两种不同量纲的分数归一化，稳定可靠。
# ======================================================================
class HybridRRFRetriever(BaseRetriever):
    vectorstore: Any = Field(exclude=True)
    pages: List[Document] = Field(default_factory=list)
    bm25: Any = Field(default=None, exclude=True)
    k_vector: int = VECTOR_K
    k_bm25: int = BM25_K
    k_final: int = FINAL_K
    rrf_c: int = RRF_C
    bm25_weight: float = BM25_WEIGHT
    use_hybrid: bool = ENABLE_HYBRID

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vec_docs = _dedupe(self.vectorstore.similarity_search(query, k=self.k_vector))

        if not self.use_hybrid or self.bm25 is None:
            return vec_docs[: self.k_final]

        scores = self.bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.k_bm25]
        bm_docs = _dedupe([self.pages[i] for i in order if scores[i] > 0])

        # RRF（倒数排名融合）
        # 关键：每个排序列表都必须【先按内容去重再算名次】。
        # 否则同一个块在列表里出现两次时会各记一次分（1/(c+1) + 1/(c+2)），
        # 人为抬高分数，把仅靠另一路命中的正确答案挤出前 N。
        fused: dict = {}
        for docs, weight in ((vec_docs, 1.0), (bm_docs, self.bm25_weight)):
            for rank, doc in enumerate(docs, 1):
                key = doc.page_content
                if key not in fused:
                    fused[key] = [0.0, doc]
                fused[key][0] += weight / (self.rrf_c + rank)

        ranked = sorted(fused.values(), key=lambda x: -x[0])
        return [doc for _, doc in ranked][: self.k_final]


def _dedupe(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


def build_hybrid_retriever(vectorstore) -> HybridRRFRetriever:
    """从向量库中取出全部文本块，建立 BM25 索引，组装混合检索器。"""
    data = vectorstore.get(include=["documents", "metadatas"])
    texts = data.get("documents") or []
    metas = data.get("metadatas") or [{}] * len(texts)

    seen, pages = set(), []
    for t, m in zip(texts, metas):
        if t in seen:
            continue
        seen.add(t)
        pages.append(Document(page_content=t, metadata=m or {}))

    bm25 = SimpleBM25([p.page_content for p in pages]) if pages else None
    print(f"[检索] 向量库共 {len(texts)} 块，去重后 {len(pages)} 块参与混合检索 "
          f"(vector_k={VECTOR_K}, bm25_k={BM25_K}, final_k={FINAL_K}, bm25_weight={BM25_WEIGHT}, hybrid={ENABLE_HYBRID})")
    if len(texts) != len(pages):
        print(f"[检索][警告] 向量库存在 {len(texts) - len(pages)} 条重复块！"
              f"请在 Streamlit 上重新点击“构建向量库”（会先清空旧库）。")

    return HybridRRFRetriever(vectorstore=vectorstore, pages=pages, bm25=bm25)


def build_rag_chain(vectorstore):
    # 1. 创建独立的 HTTP 客户端和 LLM
    http_client = httpx.Client()
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        http_client=http_client,
    )

    # 2. 混合检索器（向量 k=15 + BM25 k=15 -> RRF 融合取 final_k=10）
    retriever = build_hybrid_retriever(vectorstore)

    # 2.1 可选：MultiQueryRetriever（默认关闭）
    #     让 LLM 把问题改写成多个 query 再分别检索，能提升召回，但每次提问
    #     会多花 1 次 API 调用、增加约 1~3 秒延迟。切换 config.ENABLE_MULTI_QUERY。
    if ENABLE_MULTI_QUERY:
        # langchain 1.x 把 Retriever 系列挪到了 langchain_classic，这里做双路径兼容
        try:
            from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        except ModuleNotFoundError:
            from langchain.retrievers.multi_query import MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        print("[检索] 已启用 MultiQueryRetriever")

    # 3. 构建 RAG 链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
