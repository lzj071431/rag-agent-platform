import os
import gc
import tempfile
import httpx
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List

# ================= 原有的模块导入 =================
from loader import load_documents
from splitter import split_documents
from vector_store import build_vectorstore, load_vectorstore
from rag_chain import build_rag_chain
from config import PERSIST_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

# ================= CrewAI 竞品分析模块 =================
from crewai_analysis import run_competitor_analysis, reset_vectorstore_cache

# ================= LangGraph 相关的导入 =================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# ================= 页面配置 =================
st.set_page_config(page_title="AI 应用平台", layout="wide")
st.title("🤖 AI 智能应用平台")
st.caption("集成了 RAG 智能问答与自动周报生成 Agent")

# ================= 侧边栏：文档管理（仅用于RAG） =================
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 或 TXT", type=["pdf", "txt"])
    if st.button("构建向量库"):
        if uploaded_file is not None:
            with st.spinner("正在解析文档并构建向量库..."):
                # ---- 先释放并清空旧的向量库引用 ----
                # 关键：Streamlit 会一直持有 session_state.vectorstore，
                # 它锁住 chroma.sqlite3 导致旧库删不掉、新数据被"追加"成重复块。
                # 这里先用它自己的连接删掉集合（数据库级操作，不受文件锁影响），
                # 再清空缓存，保证下面重建时是干净状态。
                old_store = st.session_state.pop("vectorstore", None)
                st.session_state.pop("rag_chain", None)
                st.session_state.pop("retriever", None)
                if old_store is not None:
                    try:
                        old_store.delete_collection()
                        print("[构建向量库] 已删除旧集合（释放重复数据）")
                    except Exception as e:
                        print(f"[构建向量库] 旧集合删除跳过: {type(e).__name__}")
                gc.collect()

                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                docs = load_documents(tmp_path)
                # 用真实文件名覆盖临时文件路径，方便在"引用来源"里辨认
                for d in docs:
                    d.metadata["source"] = uploaded_file.name

                chunks = split_documents(docs)

                # ---- 终端打印切分后 chunk 总数（排障用）----
                print(f"\n[构建向量库] 文件={uploaded_file.name}  "
                      f"文档数={len(docs)}  切分后 chunk 总数={len(chunks)}\n")

                # reset=True：先清目录（清不掉则删集合兜底），避免重复追加（原实现会累加）
                build_vectorstore(chunks, PERSIST_DIR, reset=True)
                reset_vectorstore_cache()   # 让 Tab 3 的 CrewAI 工具改用新库，而不是旧连接

                st.session_state.vectorstore = load_vectorstore(PERSIST_DIR)

                st.success(f"向量库构建完成，共 {len(chunks)} 个文本块")
                os.unlink(tmp_path)
        else:
            st.warning("请先上传文件")

# 加载向量库到 session_state（如果存在）
if "vectorstore" not in st.session_state:
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.session_state.vectorstore = load_vectorstore(PERSIST_DIR)


# ================= 定义周报 Agent 的状态和节点 =================
class WeeklyReportState(TypedDict):
    tasks: List[dict]
    stats: dict
    report: str
    status: str


def fetch_tasks(state: WeeklyReportState) -> WeeklyReportState:
    mock_tasks = [
        {"id": 1, "title": "完成 RAG 系统评估", "status": "done"},
        {"id": 2, "title": "优化 Reranker 集成", "status": "done"},
        {"id": 3, "title": "编写 README 文档", "status": "done"},
        {"id": 4, "title": "调研 LangGraph 工作流", "status": "in_progress"},
        {"id": 5, "title": "准备技术分享 PPT", "status": "todo"},
    ]
    return {"tasks": mock_tasks, "status": "tasks_fetched"}


def analyze_tasks(state: WeeklyReportState) -> WeeklyReportState:
    tasks = state["tasks"]
    total = len(tasks)
    done = [t for t in tasks if t["status"] == "done"]
    in_progress = [t for t in tasks if t["status"] == "in_progress"]
    todo = [t for t in tasks if t["status"] == "todo"]
    completion_rate = len(done) / total * 100 if total > 0 else 0

    stats = {
        "total": total, "done": len(done), "in_progress": len(in_progress),
        "todo": len(todo), "completion_rate": round(completion_rate, 1),
        "done_list": [t["title"] for t in done],
        "in_progress_list": [t["title"] for t in in_progress],
        "todo_list": [t["title"] for t in todo],
    }
    return {"stats": stats, "status": "analyzed"}


def generate_report(state: WeeklyReportState) -> WeeklyReportState:
    stats = state["stats"]
    prompt = ChatPromptTemplate.from_template("""
你是一个专业的周报助手。请根据以下数据，生成一份简洁、专业的周报：
本周任务总数：{total}
已完成：{done} 项
进行中：{in_progress} 项
未开始：{todo} 项
完成率：{completion_rate}%
已完成任务列表：{done_list}
进行中任务列表：{in_progress_list}
未开始任务列表：{todo_list}
请用 Markdown 格式输出，包含：本周工作总结、完成情况、下周计划。语言要专业、积极。
""")

    # 注意：这里为了避免每次重跑都新建 httpx.Client，我们直接用全局的 http_client（如果有的话，或者直接实例化）
    http_client = httpx.Client()
    llm = ChatOpenAI(model=MODEL_NAME, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, temperature=0.3,
                     http_client=http_client)

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "total": stats["total"], "done": stats["done"], "in_progress": stats["in_progress"],
        "todo": stats["todo"], "completion_rate": stats["completion_rate"],
        "done_list": "\n".join(f"- {t}" for t in stats["done_list"]),
        "in_progress_list": "\n".join(f"- {t}" for t in stats["in_progress_list"]),
        "todo_list": "\n".join(f"- {t}" for t in stats["todo_list"]),
    })
    return {"report": report, "status": "report_generated"}


# 构建 LangGraph 工作流（放在函数外，只需构建一次）
workflow = StateGraph(WeeklyReportState)
workflow.add_node("fetch_tasks", fetch_tasks)
workflow.add_node("analyze_tasks", analyze_tasks)
workflow.add_node("generate_report", generate_report)
workflow.set_entry_point("fetch_tasks")
workflow.add_edge("fetch_tasks", "analyze_tasks")
workflow.add_edge("analyze_tasks", "generate_report")
workflow.add_edge("generate_report", END)
report_app = workflow.compile()

# ================= 主界面：使用 Tabs 分栏 =================
tab1, tab2, tab3 = st.tabs(["📄 智能文档问答", "📊 自动周报生成", "🤖 竞品分析 Agent"])

# ---------------- Tab 1: 智能文档问答 ----------------
with tab1:
    if "vectorstore" in st.session_state:
        # 缓存 RAG 链，避免每次提问都重建
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(st.session_state.vectorstore)

        question = st.text_input("输入你的问题：", key="rag_question")
        if question:
            with st.spinner("检索并生成回答..."):
                answer = st.session_state.rag_chain.invoke(question)
                source_docs = st.session_state.retriever.invoke(question)

            st.markdown("### 回答")
            st.write(answer)

            with st.expander("查看引用来源"):
                for i, doc in enumerate(source_docs, 1):
                    st.markdown(f"**来源 {i}**")
                    st.text(doc.page_content[:500] + "...")
                    meta_bits = []
                    if "source" in doc.metadata:
                        meta_bits.append(f"文件：{doc.metadata['source']}")
                    if doc.metadata.get("section"):
                        meta_bits.append(f"章节：{doc.metadata['section']}")
                    if doc.metadata.get("chunk_id"):
                        meta_bits.append(f"块ID：{doc.metadata['chunk_id']}")
                    if meta_bits:
                        st.caption(" ｜ ".join(meta_bits))
    else:
        st.info("👈 请先在左侧上传文档并构建向量库")

# ---------------- Tab 2: 自动周报生成 ----------------
with tab2:
    st.markdown("### 📝 一键生成本周工作周报")
    st.markdown("点击下方按钮，Agent 将自动获取本周任务、统计完成情况，并调用 DeepSeek 生成专业周报。")

    if st.button("🚀 生成周报", type="primary"):
        with st.status("正在生成周报...", expanded=True) as status:
            st.write("📥 正在获取本周任务...")
            # 运行 LangGraph 工作流
            final_state = report_app.invoke({"tasks": [], "stats": {}, "report": "", "status": "start"})
            st.write("📊 统计分析完成...")
            st.write("✍️ 正在调用大模型生成报告...")
            status.update(label="✅ 周报生成完成！", state="complete", expanded=False)

        st.markdown("---")
        st.markdown(final_state["report"])

# ---------------- Tab 3: 竞品分析 Agent ----------------
with tab3:
    st.markdown("### 🤖 CrewAI 竞品分析 Agent")
    st.markdown(
        "由三个 Agent 依次协作完成：**资深竞品调研员**（调用 RAG 工具检索本地知识库）"
        " → **市场分析专家** → **专业商业报告撰写人**。"
        "所有结论均标注来源，知识库中没有的内容会如实写明\"知识库中未找到\"。"
    )

    col1, col2 = st.columns(2)
    with col1:
        product = st.text_input("分析产品：", value="妙可蓝多", key="crew_product")
    with col2:
        competitor = st.text_input("竞品名称：", value="伊利", key="crew_competitor")

    st.info("⏳ 该功能包含多个 Agent 协作，预计耗时 1-2 分钟，请耐心等待")

    if st.button("🚀 生成竞品分析报告", type="primary", key="crew_run"):
        if not product.strip() or not competitor.strip():
            st.warning("请先填写「分析产品」和「竞品名称」")
        else:
            with st.status("正在调用 CrewAI 多 Agent 协作...", expanded=True) as status:
                st.write("📚 正在加载本地知识库...")
                st.write(f"🔍 调研员检索 → 📊 分析师对比 → ✍️ 撰稿人撰写（{product} vs {competitor}）")
                report = run_competitor_analysis(product.strip(), competitor.strip(), verbose=False)
                st.session_state["crew_report"] = report
                status.update(label="✅ 竞品分析报告生成完成！", state="complete", expanded=False)

    # 报告存在 session_state 里，这样页面重跑（点别处、切 Tab）也不会丢
    if st.session_state.get("crew_report"):
        st.markdown("---")
        st.markdown(st.session_state["crew_report"])
        st.download_button(
            "⬇️ 下载 Markdown 报告",
            st.session_state["crew_report"],
            file_name="competitor_analysis_report.md",
            mime="text/markdown",
        )