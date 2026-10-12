import os
import logging
import httpx
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

# ================= 1. 初始化与日志配置 =================
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 从环境变量读取飞书 Webhook，避免硬编码导致泄露
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK_URL", "")
MAX_CONTENT = 20_000  # 飞书单条消息字符上限，留余量

# ================= 2. 定义状态 =================
class WeeklyReportState(TypedDict):
    tasks: List[dict]
    stats: dict
    report: str
    status: str
    error: str  # 关键：必须声明，否则 LangGraph 更新状态时会报错

# ================= 3. 初始化 LLM =================
http_client = httpx.Client()
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
    http_client=http_client,
)

# ================= 4. 定义节点函数 =================

def fetch_tasks(state: WeeklyReportState) -> WeeklyReportState:
    """节点1：获取本周任务（模拟数据）"""
    mock_tasks = [
        {"id": 1, "title": "完成 RAG 系统评估", "status": "done"},
        {"id": 2, "title": "优化 Reranker 集成", "status": "done"},
        {"id": 3, "title": "编写 README 文档", "status": "done"},
        {"id": 4, "title": "调研 LangGraph 工作流", "status": "in_progress"},
        {"id": 5, "title": "准备技术分享 PPT", "status": "todo"},
    ]
    logger.info("📥 [节点1] 获取本周任务...")
    return {"tasks": mock_tasks, "status": "tasks_fetched"}

def analyze_tasks(state: WeeklyReportState) -> WeeklyReportState:
    """节点2：分析任务完成情况"""
    tasks = state["tasks"]
    total = len(tasks)
    done = [t for t in tasks if t["status"] == "done"]
    in_progress = [t for t in tasks if t["status"] == "in_progress"]
    todo = [t for t in tasks if t["status"] == "todo"]
    completion_rate = len(done) / total * 100 if total > 0 else 0

    stats = {
        "total": total,
        "done": len(done),
        "in_progress": len(in_progress),
        "todo": len(todo),
        "completion_rate": round(completion_rate, 1),
        "done_list": [t["title"] for t in done],
        "in_progress_list": [t["title"] for t in in_progress],
        "todo_list": [t["title"] for t in todo],
    }
    logger.info(f"📊 [节点2] 统计完成：共 {total} 项，完成 {len(done)} 项，完成率 {completion_rate:.1f}%")
    return {"stats": stats, "status": "analyzed"}

def generate_report(state: WeeklyReportState) -> WeeklyReportState:
    """节点3：调用 LLM 生成周报"""
    stats = state["stats"]
    prompt = ChatPromptTemplate.from_template("""
你是一个专业的周报助手。请根据以下数据，生成一份简洁、专业的周报：

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
""")
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "total": stats["total"],
        "done": stats["done"],
        "in_progress": stats["in_progress"],
        "todo": stats["todo"],
        "completion_rate": stats["completion_rate"],
        "done_list": "\n".join(f"- {t}" for t in stats["done_list"]),
        "in_progress_list": "\n".join(f"- {t}" for t in stats["in_progress_list"]),
        "todo_list": "\n".join(f"- {t}" for t in stats["todo_list"]),
    })
    logger.info("✍️ [节点3] 周报生成完成")
    return {"report": report, "status": "report_generated"}

def _build_card(report: str) -> dict:
    """构建飞书消息卡片"""
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "📊 本周工作周报"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": report[:MAX_CONTENT]}],
        },
    }

def send_report(state: WeeklyReportState) -> WeeklyReportState:
    """节点4：发送周报到飞书。失败不抛异常，通过 status 向下游传递结果。"""
    if not FEISHU_WEBHOOK:
        raise RuntimeError("未配置环境变量 FEISHU_WEBHOOK_URL，请检查 .env 文件")

    report = (state.get("report") or "").strip()
    if not report:
        logger.warning("周报内容为空，跳过发送")
        return {"status": "skipped", "error": "empty report"}

    logger.info("📤 [节点4] 正在发送周报到飞书...")
    try:
        resp = httpx.post(
            FEISHU_WEBHOOK,
            json=_build_card(report),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        body = resp.json()
    except httpx.TimeoutException:
        logger.exception("发送飞书超时")
        return {"status": "failed", "error": "timeout"}
    except (httpx.HTTPError, ValueError) as e:
        logger.exception("发送飞书异常")
        return {"status": "failed", "error": str(e)}

    # 关键：以响应体 code 为准，而不是只看 HTTP 状态码
    if resp.status_code == 200 and body.get("code") == 0:
        logger.info("✅ 周报已成功发送到飞书群")
        return {"status": "sent"}

    logger.error("❌ 飞书发送失败 http=%s body=%s", resp.status_code, body)
    return {"status": "failed", "error": f"{body.get('code')}: {body.get('msg')}"}

# ================= 5. 构建 LangGraph 工作流 =================
workflow = StateGraph(WeeklyReportState)

workflow.add_node("fetch_tasks", fetch_tasks)
workflow.add_node("analyze_tasks", analyze_tasks)
workflow.add_node("generate_report", generate_report)
workflow.add_node("send_report", send_report)

workflow.set_entry_point("fetch_tasks")

workflow.add_edge("fetch_tasks", "analyze_tasks")
workflow.add_edge("analyze_tasks", "generate_report")
workflow.add_edge("generate_report", "send_report")
workflow.add_edge("send_report", END)

app = workflow.compile()

# ================= 6. 运行 =================
if __name__ == "__main__":
    logger.info("🚀 启动自动周报生成 Agent...\n")
    final_state = app.invoke({"tasks": [], "stats": {}, "report": "", "status": "start", "error": ""})
    logger.info("✅ 周报生成流程结束！")
    if final_state.get("status") == "failed":
        logger.error(f"流程中存在失败节点: {final_state.get('error')}")
    else:
        logger.info(f"最终状态: {final_state.get('status')}")