import json
from vector_store import load_vectorstore
from rag_chain import build_rag_chain


def load_test_set(path="test_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    test_set = load_test_set()
    print(f"成功加载测试集，共 {len(test_set)} 个问题。\n")

    # 加载向量库和 RAG 链
    vectorstore = load_vectorstore()
    rag_chain, retriever = build_rag_chain(vectorstore)

    n = len(test_set)
    hit_count = 0
    mrr_sum = 0.0

    print("=" * 60)
    print("检索层评估")
    print("=" * 60)

    for case in test_set:
        question = case["question"]
        relevant_ids = case["relevant_chunk_ids"]

        # 检索 Top-5 文档
        retrieved_docs = retriever.invoke(question)

        # 计算 Hit Rate 和 MRR
        hit = False
        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc.metadata.get("chunk_id", "")
            if doc_id in relevant_ids:
                hit = True
                mrr_sum += 1.0 / rank
                break

        if hit:
            hit_count += 1

        retrieved_ids = [doc.metadata.get("chunk_id") for doc in retrieved_docs]
        print(f"[{case['id']}] {question}")
        print(f"  期望命中: {relevant_ids}")
        print(f"  实际检索: {retrieved_ids}")
        print(f"  是否命中: {'✅' if hit else '❌'}\n")

    hit_rate = hit_count / n
    mrr = mrr_sum / n

    print("=" * 60)
    print(f"Hit Rate@5: {hit_rate:.2%}")
    print(f"MRR: {mrr:.4f}")
    print("=" * 60)

    # 生成层评估（打印答案供人工判断）
    print("\n" + "=" * 60)
    print("生成层评估（请人工核对答案质量）")
    print("=" * 60)

    for case in test_set:
        question = case["question"]
        answer = rag_chain.invoke(question)
        print(f"[{case['id']}] {question}")
        print(f"  标准答案: {case['ground_truth']}")
        print(f"  系统回答: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    evaluate()