"""AI 智能应用平台 —— Streamlit UI 入口。

⚠️ 本文件**只负责界面编排**，不含任何业务逻辑。
所有实际工作（检索、生成、Agent 协作）都在 ``services/`` 层完成。

职责边界
--------
* 渲染页面、收集用户输入
* 调用 service 层函数
* 用 ``st.session_state`` 缓存重量级对象（向量库、RAG 服务）
* 把异常转换成用户可读的提示

启动::

    streamlit run app.py
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import streamlit as st

# 必须最先导入：完成日志与离线开关初始化，之后才能导入重量级库
from core.config import settings
from core.logger import get_logger, new_request_id, setup_logging
from core.exceptions import AppError

setup_logging()
logger = get_logger(__name__)

# ---- service 层导入（UI 只依赖这一层）----
from services.document_service import load_documents, split_documents
from services.rag_service import RAGService
from services.report_service import WeeklyReportService
from services.vector_store_service import (
    build_vectorstore,
    delete_collection_of,
    load_vectorstore,
    vectorstore_exists,
)
from services.agent_service import CompetitorAnalysisService, reset_vectorstore_cache


# ======================================================================
# 页面配置
# ======================================================================
st.set_page_config(page_title=settings.app_name, layout="wide")
st.title("🤖 AI 智能应用平台")
st.caption("集成了 RAG 智能问答与自动周报生成 Agent")


# ======================================================================
# 会话状态管理
# ======================================================================
def _reset_rag_state() -> None:
    """清理与旧向量库绑定的会话对象。

    关键：Streamlit 会一直持有 ``session_state.vectorstore``，
    它锁住 chroma.sqlite3 导致旧库删不掉、新数据被"追加"成重复块。
    """
    old_store = st.session_state.pop("vectorstore", None)
    st.session_state.pop("rag_service", None)
    st.session_state.pop("retriever", None)

    if old_store is not None:
        # 用实例自己的连接删掉集合（数据库级操作，不受文件锁影响）
        delete_collection_of(old_store)
    gc.collect()


def _ensure_vectorstore_loaded() -> None:
    """若本地已有向量库且会话中尚未加载，则加载到 session_state。"""
    if "vectorstore" in st.session_state:
        return
    if not vectorstore_exists():
        return
    try:
        st.session_state.vectorstore = load_vectorstore()
        logger.info("已从本地加载向量库 | dir=%s", settings.persist_dir)
    except AppError:
        logger.exception("向量库加载失败，等待用户重新构建")
        st.session_state.pop("vectorstore", None)


def _get_rag_service() -> RAGService | None:
    """获取（并缓存）RAG 服务。"""
    if "rag_service" in st.session_state:
        return st.session_state.rag_service
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return None
    st.session_state.rag_service = RAGService(vectorstore)
    return st.session_state.rag_service


# ======================================================================
# 侧边栏：知识库管理
# ======================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("📁 知识库管理")
        uploaded_file = st.file_uploader("上传 PDF 或 TXT", type=["pdf", "txt"])

        if st.button("构建向量库"):
            if uploaded_file is None:
                st.warning("请先上传文件")
                return

            with st.spinner("正在解析文档并构建向量库..."):
                try:
                    _build_knowledge_base(uploaded_file)
                except AppError as exc:
                    logger.exception("知识库构建失败")
                    st.error(f"构建失败：{exc}")
                except Exception as exc:  # noqa: BLE001 - UI 兜底
                    logger.exception("知识库构建出现未预期错误")
                    st.error(f"构建失败：{type(exc).__name__}: {exc}")


def _build_knowledge_base(uploaded_file) -> None:
    """处理上传文件 → 切分 → 建库 → 刷新会话缓存。"""
    new_request_id()
    logger.info("开始构建知识库 | file=%s", uploaded_file.name)

    # 1) 释放旧向量库引用（否则 Windows 下 sqlite 被锁，删不掉旧库）
    _reset_rag_state()

    # 2) 落临时文件（保留真实扩展名，loader 依赖后缀判断类型）
    suffix = Path(uploaded_file.name).suffix.lower() or ".txt"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # 3) 加载 + 切分
        docs = load_documents(tmp_path)
        # 用真实文件名覆盖临时文件路径，方便在"引用来源"里辨认
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        chunks = split_documents(docs)
        logger.info(
            "文档处理完成 | file=%s | 文档数=%d | chunk 总数=%d",
            uploaded_file.name,
            len(docs),
            len(chunks),
        )

        # 4) 建库（reset=True 保证幂等，不会累加重复块）
        build_vectorstore(chunks, reset=True)

        # 5) 让 CrewAI 工具改用新库，而不是旧连接
        reset_vectorstore_cache()

        # 6) 刷新会话缓存
        st.session_state.vectorstore = load_vectorstore()
        st.session_state.pop("rag_service", None)

        st.success(f"向量库构建完成，共 {len(chunks)} 个文本块")
        logger.info("知识库构建成功 | file=%s | chunks=%d", uploaded_file.name, len(chunks))

    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                logger.warning("临时文件清理失败 | path=%s", tmp_path)


# ======================================================================
# Tab 1: 智能文档问答
# ======================================================================
def render_rag_tab() -> None:
    if "vectorstore" not in st.session_state:
        st.info("👈 请先在左侧上传文档并构建向量库")
        return

    # 构造 RAGService 会加载 Embedding 模型并校验 API Key，可能抛
    # ConfigurationError / VectorStoreError。必须在 UI 层捕获，
    # 否则会以裸 traceback 形式炸掉整个 Tab。
    try:
        service = _get_rag_service()
    except AppError as exc:
        logger.exception("RAG 服务初始化失败")
        st.error(f"问答服务初始化失败：{exc}")
        return
    except Exception as exc:  # noqa: BLE001 - UI 兜底
        logger.exception("RAG 服务初始化出现未预期错误")
        st.error(f"问答服务初始化失败：{type(exc).__name__}: {exc}")
        return

    if service is None:
        st.info("👈 请先在左侧上传文档并构建向量库")
        return

    question = st.text_input("输入你的问题：", key="rag_question")
    if not question:
        return

    with st.spinner("检索并生成回答..."):
        try:
            result = service.ask(question)
        except AppError as exc:
            logger.exception("问答失败")
            st.error(f"问答失败：{exc}")
            return
        except Exception as exc:  # noqa: BLE001 - UI 兜底
            logger.exception("问答出现未预期错误")
            st.error(f"问答失败：{type(exc).__name__}: {exc}")
            return

    st.markdown("### 回答")
    st.write(result.answer)

    if not result.retrieval_hit:
        st.warning("未在知识库中检索到相关内容，建议换个问法或补充文档。")
        return

    with st.expander(f"查看引用来源（{len(result.sources)} 块，耗时 {result.elapsed_seconds:.2f}s）"):
        for i, item in enumerate(result.source_metadata(), 1):
            st.markdown(f"**来源 {i}**")
            st.text(item["preview"] + "...")
            meta_bits = [f"文件：{item['source']}"]
            if item.get("section"):
                meta_bits.append(f"章节：{item['section']}")
            if item.get("chunk_id"):
                meta_bits.append(f"块ID：{item['chunk_id']}")
            st.caption(" ｜ ".join(meta_bits))


# ======================================================================
# Tab 2: 自动周报生成
# ======================================================================
def render_weekly_report_tab() -> None:
    st.markdown("### 📝 一键生成本周工作周报")
    st.markdown(
        "点击下方按钮，Agent 将自动获取本周任务、统计完成情况，"
        "并调用 DeepSeek 生成专业周报。"
    )

    if not st.button("🚀 生成周报", type="primary"):
        # 页面重跑后仍展示上一次结果
        cached = st.session_state.get("weekly_report")
        if cached:
            st.markdown("---")
            st.markdown(cached)
        return

    with st.status("正在生成周报...", expanded=True) as status:
        try:
            st.write("📥 正在获取本周任务...")
            service = WeeklyReportService(send_to_feishu=False)
            st.write("📊 统计分析完成...")
            st.write("✍️ 正在调用大模型生成报告...")
            result = service.run()
            st.session_state["weekly_report"] = result.report
            status.update(label="✅ 周报生成完成！", state="complete", expanded=False)
        except AppError as exc:
            logger.exception("周报生成失败")
            status.update(label="❌ 周报生成失败", state="error", expanded=True)
            st.error(f"生成失败：{exc}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("周报生成出现未预期错误")
            status.update(label="❌ 周报生成失败", state="error", expanded=True)
            st.error(f"生成失败：{type(exc).__name__}: {exc}")
            return

    st.markdown("---")
    st.markdown(result.report)


# ======================================================================
# Tab 3: 竞品分析 Agent
# ======================================================================
def render_competitor_tab() -> None:
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
                try:
                    st.write("📚 正在加载本地知识库...")
                    st.write(
                        f"🔍 调研员检索 → 📊 分析师对比 → ✍️ 撰稿人撰写"
                        f"（{product} vs {competitor}）"
                    )
                    service = CompetitorAnalysisService()
                    report = service.run(product.strip(), competitor.strip(), verbose=False)
                    st.session_state["crew_report"] = report.content
                    status.update(
                        label="✅ 竞品分析报告生成完成！", state="complete", expanded=False
                    )
                except AppError as exc:
                    logger.exception("竞品分析失败")
                    status.update(label="❌ 竞品分析失败", state="error", expanded=True)
                    st.error(f"分析失败：{exc}")
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.exception("竞品分析出现未预期错误")
                    status.update(label="❌ 竞品分析失败", state="error", expanded=True)
                    st.error(f"分析失败：{type(exc).__name__}: {exc}")
                    return

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


# ======================================================================
# 主流程
# ======================================================================
def main() -> None:
    new_request_id()
    _ensure_vectorstore_loaded()

    # 配置自检：缺 Key 时提前提示，而不是等用户点按钮才炸
    if not settings.has_llm_credentials:
        st.warning(
            "⚠️ 未检测到 DEEPSEEK_API_KEY，大模型相关功能将不可用。"
            "请在项目根目录的 .env 中配置后重启应用。"
        )

    render_sidebar()

    tab1, tab2, tab3 = st.tabs(
        ["📄 智能文档问答", "📊 自动周报生成", "🤖 竞品分析 Agent"]
    )
    with tab1:
        render_rag_tab()
    with tab2:
        render_weekly_report_tab()
    with tab3:
        render_competitor_tab()


main()
