"""
CrewAI 多 Agent 竞品分析 —— 接入本地 RAG 知识库（ChromaDB）

两种用法：
1. 命令行：python crewai_analysis.py
2. 被 app.py 作为第三个 Tab 调用：run_competitor_analysis(product, competitor)
"""

import os
import threading

# ⚠️ 必须在导入 crewai / transformers 之前设置离线开关。
#    config.py 里也设了同样的变量，但 crewai 会先于 config.py 被导入，
#    所以这里要提前设一遍，否则 HuggingFaceEmbeddings 会尝试联网下载模型。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 关掉遥测上报：Chroma 会往 posthog、CrewAI 会往 telemetry.crewai.com 发数据，
# 连不上时层层重试退避，既刷屏又拖慢请求（在网页端这几秒是白等的）。
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")     # Chroma 的产品遥测
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "1")     # CrewAI 的 OTEL 遥测

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# ================= 环境变量：绝对路径强制加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(f"❌ 未找到 DEEPSEEK_API_KEY！请检查 {env_path}")

# ================= 初始化 LLM（DeepSeek 的 OpenAI 兼容接口） =================
deepseek_llm = LLM(
    model="deepseek-chat",
    custom_openai=True,
    base_url="https://api.deepseek.com/v1",
    api_key=api_key,
    temperature=0.3,
)

# ================= 封装 RAG 检索为 CrewAI Tool =================
from vector_store import load_vectorstore

# 向量库用绝对路径，避免"必须在某个工作目录下运行"的隐式依赖
CHROMA_DIR = os.path.join(current_dir, "chroma_db")

# bge-small-zh-v1.5 + Chroma 默认 L2 距离，在本知识库上实测：
#   相关问题最优距离 ≈ 0.38 ~ 1.00；无关问题（如"伊利""蒙牛"）≥ 1.16。
# 取 1.05 作为分界线：只有距离小于它的片段才认为是真正检索到的内容，
# 否则返回"未找到"，从源头堵住"检索到一堆无关片段、再编造答案"的路径。
RELEVANCE_DISTANCE_THRESHOLD = 1.05

REPORT_PATH = os.path.join(current_dir, "competitor_analysis_report.md")

_vectorstore = None
_vectorstore_lock = threading.Lock()


def _get_vectorstore():
    """惰性加载并缓存向量库，避免每次调用工具都重新加载 embedding 模型。

    CrewAI 可能在一个 Agent 内并行发起多次工具调用，如果两个线程同时初始化，
    Chroma 会因争抢同一个 persist 目录的连接而报
    "Could not connect to tenant default_tenant"，因此这里必须加锁。
    """
    global _vectorstore
    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                _vectorstore = load_vectorstore(persist_dir=CHROMA_DIR)
    return _vectorstore


def reset_vectorstore_cache():
    """丢弃已缓存的向量库连接。侧边栏重建知识库后调用，避免继续查旧的库。"""
    global _vectorstore
    with _vectorstore_lock:
        _vectorstore = None


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """从本地财务报告知识库中检索相关信息。输入一个完整的搜索查询语句，返回最相关的文档片段。

    Args:
        query: 完整的搜索查询语句，例如"妙可蓝多2025年股票期权的行权价格"。
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=5)
    docs = [doc for doc, score in results if score <= RELEVANCE_DISTANCE_THRESHOLD]
    if not docs:
        return "知识库中未找到相关信息。"
    return "\n\n".join(
        f"[来源 {i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def _build_crew(product: str, competitor: str, verbose: bool) -> Crew:
    """每次调用都新建 Agent / Task / Crew。

    不复用模块级单例：Streamlit 每次交互都会重新执行脚本，复用同一个 Agent
    会把上一轮的执行状态带进来，导致跨次运行互相污染。
    """
    researcher = Agent(
        role='资深竞品调研员',
        goal=f'基于本地知识库中的财务报告，全面搜集关于 {product} 及其竞品 {competitor} 的功能、定价、用户评价等信息',
        backstory=(
            '你是一位有着10年经验的市场调研员。你非常擅长使用 search_knowledge_base 工具，'
            '从海量的财务报告和行业文件中快速定位关键数据，并以结构化的形式整理出来。'
            '你只相信知识库中检索到的事实，绝不凭空编造：'
            '凡是知识库中没有检索到的内容，你都会如实标注"知识库中未找到"。'
        ),
        llm=deepseek_llm,
        tools=[search_knowledge_base],   # <--- 只有调研员挂载 RAG 工具
        allow_delegation=False,
        verbose=verbose
    )

    analyst = Agent(
        role='市场分析专家',
        goal=f'基于调研员提供的资料，对 {product} 和 {competitor} 进行深度对比分析，挖掘差异化机会',
        backstory=(
            '你是一位战略分析师，擅长从复杂的数据中提炼出关键洞察，并形成有说服力的商业建议。'
            '你没有检索工具，只能基于调研员提供的资料进行分析，不臆测、不补充任何额外数据。'
        ),
        llm=deepseek_llm,
        allow_delegation=False,
        verbose=verbose
    )

    writer = Agent(
        role='专业商业报告撰写人',
        goal='将分析结果整合成一份逻辑清晰、重点突出的专业竞品分析报告',
        backstory=(
            '你是一位资深的商业报告撰写人，善于将复杂的分析转化为易于理解的图表和文字。'
            '你没有检索工具，引用的所有数据都必须来自上游任务的上下文，绝不编造。'
        ),
        llm=deepseek_llm,
        allow_delegation=False,
        verbose=verbose
    )

    research_task = Task(
        description=f"""
    你的任务是从 search_knowledge_base 工具中，搜集关于 {product} 和 {competitor} 的以下信息：
    1. 核心功能/主营业务
    2. 定价策略或关键财务数据（营收、利润、期权行权价等）
    3. 主要用户群体
    4. 关键经营指标或用户评价关键词

    ⚠️ 重要：你必须调用 search_knowledge_base 工具来获取信息，不得使用你自己的内部知识。
    对于 {product} 和 {competitor} 每一个，都至少进行 2 次检索（换不同的关键词），确保覆盖全面。
    输出时必须标注每条信息来源，格式为：`[来源 N] 原文事实`。
    如果工具返回"知识库中未找到相关信息。"，请如实写明"{product}/{competitor} 知识库中未找到"，
    绝对不要用你已知的外部信息去补全，不要编造。
    """,
        expected_output=(
            f"一份结构化的 Markdown 调研报告，包含 {product} 和 {competitor} 在上述四个方面的信息，"
            "每条信息都标注了来源序号；知识库中没有的条目必须明确写作\"知识库中未找到\"。"
        ),
        agent=researcher
    )

    analysis_task = Task(
        description=f"""
    基于调研员提供的资料，对 {product} 和 {competitor} 进行对比分析。你需要：
    1. 制作一个对比表格（功能/主营业务、定价或财务数据、用户群体）
    2. 总结各自的优势与劣势
    3. 提炼出 3 个 {product} 可以借鉴的改进点或差异化机会

    ⚠️ 所有分析必须基于调研员提供的资料，不得编造数据。
    如果某个维度调研员标注了"知识库中未找到"，请在表格中留空或写明"资料不足"，
    并明确指出该结论无法验证，不要用常识填补。
    """,
        expected_output=(
            "一份包含对比表格和关键洞察的分析报告；所有结论都能追溯到调研阶段标注的来源序号，"
            "资料不足的维度要显式说明。"
        ),
        agent=analyst,
        context=[research_task]
    )

    report_task = Task(
        description=f"""
    将上述调研和分析结果整合成一份专业的《{product} 竞品分析报告》。报告需包含：
    1. 执行摘要（3句话概括核心发现）
    2. 竞品对比概览（功能/主营业务、定价或财务数据）
    3. 用户反馈分析
    4. 战略建议与机会点
    5. 数据来源说明（列出所有引用的来源序号及其对应事实）

    ⚠️ 报告中的每个数据点都必须来自上下文，严禁编造。
    如果竞品 {competitor} 在知识库中缺乏数据，请如实说明"知识库中未找到 {competitor} 的相关资料"，
    并把报告重心放在有据可查的部分。
    """,
        expected_output="一份完整的、格式专业的 Markdown 竞品分析报告，内容真实可溯源。",
        agent=writer,
        context=[research_task, analysis_task],
        output_file=REPORT_PATH
    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, report_task],
        process=Process.sequential,   # researcher -> analyst -> writer
        verbose=verbose
    )


def run_competitor_analysis(product: str, competitor: str, verbose: bool = True) -> str:
    """执行一轮竞品分析，返回 Markdown 格式的报告字符串。

    Args:
        product: 分析的产品名，例如"妙可蓝多"。
        competitor: 竞品名，例如"伊利"。
        verbose: 是否把 Agent 协作过程打印到终端（命令行用 True，网页端用 False）。

    Returns:
        Markdown 报告字符串；报告同时会写入 competitor_analysis_report.md。
    """
    _get_vectorstore()   # 单线程预热，避免 Agent 并发调用工具时重复初始化向量库

    crew = _build_crew(product, competitor, verbose)
    result = crew.kickoff(inputs={'product': product, 'competitor': competitor})
    return result.raw


# ================= 命令行入口 =================
if __name__ == "__main__":
    print("📚 正在加载本地知识库...")
    print("🚀 启动竞品分析 Crew（基于本地知识库）...")
    report = run_competitor_analysis('妙可蓝多', '伊利')

    print(f"\n✅ 任务完成！报告已生成：{REPORT_PATH}")
    print(report)
