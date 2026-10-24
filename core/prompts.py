"""提示词模板集中管理。

原先提示词散落在 ``rag_chain.py``、``app.py``、``weekly_report_agent.py``
三处，难以统一迭代与 A/B 测试。此处收敛为常量，供各 service 引用。
"""

from __future__ import annotations

# ======================================================================
# RAG 问答
# ======================================================================
RAG_QA_TEMPLATE = """仅根据以下上下文回答问题。如果上下文中没有相关信息，请回答“根据提供的资料，我无法回答这个问题。”
上下文：
{context}

问题：{question}
"""

#: 检索为空时的兜底回答（不调用大模型，直接返回，省一次 API 费用）。
RAG_NO_CONTEXT_ANSWER = "根据提供的资料，我无法回答这个问题。"

# ======================================================================
# 周报生成
# ======================================================================
WEEKLY_REPORT_TEMPLATE = """你是一个专业的周报助手。请根据以下数据，生成一份简洁、专业的周报：

本周任务总数：{total}
已完成：{done} 项
进行中：{in_progress} 项
未开始：{todo} 项
完成率：{completion_rate}%

已完成任务列表：
{done_list}

进行中任务列表：
{in_progress_list}

未开始任务列表：
{todo_list}

请用 Markdown 格式输出，包含：本周工作总结、完成情况、下周计划。语言要专业、积极。
"""

# ======================================================================
# 竞品分析（CrewAI 多 Agent）
# ======================================================================
RESEARCHER_ROLE = "资深竞品调研员"
RESEARCHER_BACKSTORY = (
    "你是一位有着10年经验的市场调研员。你非常擅长使用 search_knowledge_base 工具，"
    "从海量的财务报告和行业文件中快速定位关键数据，并以结构化的形式整理出来。"
    "你只相信知识库中检索到的事实，绝不凭空编造："
    '凡是知识库中没有检索到的内容，你都会如实标注"知识库中未找到"。'
)

ANALYST_ROLE = "市场分析专家"
ANALYST_BACKSTORY = (
    "你是一位战略分析师，擅长从复杂的数据中提炼出关键洞察，并形成有说服力的商业建议。"
    "你没有检索工具，只能基于调研员提供的资料进行分析，不臆测、不补充任何额外数据。"
)

WRITER_ROLE = "专业商业报告撰写人"
WRITER_BACKSTORY = (
    "你是一位资深的商业报告撰写人，善于将复杂的分析转化为易于理解的图表和文字。"
    "你没有检索工具，引用的所有数据都必须来自上游任务的上下文，绝不编造。"
)


def build_research_task_description(product: str, competitor: str) -> str:
    """调研员任务描述（含产品名插值）。"""
    return f"""
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
"""


def build_research_expected_output(product: str, competitor: str) -> str:
    return (
        f"一份结构化的 Markdown 调研报告，包含 {product} 和 {competitor} 在上述四个方面的信息，"
        '每条信息都标注了来源序号；知识库中没有的条目必须明确写作"知识库中未找到"。'
    )


def build_analysis_task_description(product: str, competitor: str) -> str:
    """分析师任务描述。"""
    return f"""
基于调研员提供的资料，对 {product} 和 {competitor} 进行对比分析。你需要：
1. 制作一个对比表格（功能/主营业务、定价或财务数据、用户群体）
2. 总结各自的优势与劣势
3. 提炼出 3 个 {product} 可以借鉴的改进点或差异化机会

⚠️ 所有分析必须基于调研员提供的资料，不得编造数据。
如果某个维度调研员标注了"知识库中未找到"，请在表格中留空或写明"资料不足"，
并明确指出该结论无法验证，不要用常识填补。
"""


ANALYSIS_EXPECTED_OUTPUT = (
    "一份包含对比表格和关键洞察的分析报告；所有结论都能追溯到调研阶段标注的来源序号，"
    "资料不足的维度要显式说明。"
)


def build_report_task_description(product: str, competitor: str) -> str:
    """撰稿人任务描述。"""
    return f"""
将上述调研和分析结果整合成一份专业的《{product} 竞品分析报告》。报告需包含：
1. 执行摘要（3句话概括核心发现）
2. 竞品对比概览（功能/主营业务、定价或财务数据）
3. 用户反馈分析
4. 战略建议与机会点
5. 数据来源说明（列出所有引用的来源序号及其对应事实）

⚠️ 报告中的每个数据点都必须来自上下文，严禁编造。
如果竞品 {competitor} 在知识库中缺乏数据，请如实说明"知识库中未找到 {competitor} 的相关资料"，
并把报告重心放在有据可查的部分。
"""


REPORT_EXPECTED_OUTPUT = "一份完整的、格式专业的 Markdown 竞品分析报告，内容真实可溯源。"


__all__ = [
    "RAG_QA_TEMPLATE",
    "RAG_NO_CONTEXT_ANSWER",
    "WEEKLY_REPORT_TEMPLATE",
    "RESEARCHER_ROLE",
    "RESEARCHER_BACKSTORY",
    "ANALYST_ROLE",
    "ANALYST_BACKSTORY",
    "WRITER_ROLE",
    "WRITER_BACKSTORY",
    "build_research_task_description",
    "build_research_expected_output",
    "build_analysis_task_description",
    "ANALYSIS_EXPECTED_OUTPUT",
    "build_report_task_description",
    "REPORT_EXPECTED_OUTPUT",
]
