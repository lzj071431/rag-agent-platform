"""BM25 关键词检索（纯 Python 实现，零额外依赖）。

从原 ``rag_chain.py`` 中抽出的通用算法，不含任何业务语义。

为什么需要它？
--------------
财报问答里大量是"15.83""行权价格"这类精确关键词，纯向量检索对
精确数字/专有名词不敏感。加一路 BM25 能显著提升命中率，
再与向量结果用 RRF 融合（见 ``services.retrieval``）。

实现说明
--------
中文没有天然分词，这里用**字符二元组（bigram）**作为 token，
对中文财报文本效果稳定且无需引入 jieba 等分词依赖。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

#: 保留中英文、数字；其余字符（标点、空白）全部剔除。
_NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """把文本切成字符二元组列表。

    例：``"行权价格"`` → ``["行权", "权价", "价格"]``。
    文本长度不足 2 时退化为单 token，避免空列表导致后续除零。
    """
    cleaned = _NON_WORD_RE.sub("", (text or "").lower())
    grams = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    return grams or ([cleaned] if cleaned else [])


class SimpleBM25:
    """极简 BM25（Okapi）实现。

    参数量纲与经典实现一致：
    * ``k1`` 控制词频饱和度（默认 1.5）；
    * ``b`` 控制文档长度归一化强度（默认 0.75）。
    """

    __slots__ = ("k1", "b", "docs", "lens", "avg", "idf", "_n")

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[Counter[str]] = [Counter(tokenize(t)) for t in corpus]
        self.lens: List[int] = [sum(c.values()) for c in self.docs]
        # 纯标点/空语料会让所有文档长度为 0，avg 归零后 scores() 会除零。
        # 用 1.0 兜底：此时长度归一化项退化为常数，不影响排序（全为 0 分）。
        self.avg: float = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        if not self.avg:
            self.avg = 1.0
        self._n = len(self.docs)

        # 文档频率 → IDF（使用带 +1 平滑的经典公式，避免负 IDF）
        df: Counter[str] = Counter()
        for counter in self.docs:
            df.update(counter.keys())
        n = self._n
        self.idf: dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> List[float]:
        """返回查询对每个文档的 BM25 得分（与语料顺序一一对应）。"""
        if not self.docs:
            return []

        q_tokens = Counter(tokenize(query))
        if not q_tokens:
            return [0.0] * len(self.docs)

        out: List[float] = []
        for idx, counter in enumerate(self.docs):
            score = 0.0
            doc_len = self.lens[idx]
            norm = self.k1 * (1 - self.b + self.b * doc_len / self.avg)
            for term, q_freq in q_tokens.items():
                f = counter.get(term, 0)
                if f:
                    score += (
                        self.idf.get(term, 0.0)
                        * q_freq
                        * (f * (self.k1 + 1))
                        / (f + norm)
                    )
            out.append(score)
        return out

    def top_k(self, query: str, k: int, *, min_score: float = 0.0) -> List[int]:
        """返回得分最高的 ``k`` 个文档下标（按分数降序，过滤零分）。"""
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [i for i in order if scores[i] > min_score]


__all__ = ["SimpleBM25", "tokenize"]
