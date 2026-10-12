import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent  # LangChain 1.0 推荐的 Agent 入口
from vector_store import load_vectorstore
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME

load_dotenv()

# 1. 初始化 LLM
http_client = httpx.Client()
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
    http_client=http_client,
)

# 2. 加载本地的向量库（确保你之前已经构建过）
vectorstore = load_vectorstore()


# 3. 将 RAG 封装成 Tool（工具）
@tool
def search_finance_docs(query: str) -> str:
    """
    当用户询问关于财务报告、股票期权、行权价格、营收数据等具体信息时，使用此工具进行检索。
    输入应该是一个完整的搜索查询语句。
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


# 4. 创建 Agent
# create_agent 会返回一个 LangGraph 对象
agent_executor = create_agent(
    model=llm,
    tools=[search_finance_docs],  # 将你的 RAG 工具交给 Agent
    system_prompt="你是一个专业的财报分析助手。当用户询问财报相关问题时，请优先使用 search_finance_docs 工具检索信息，然后基于检索结果回答。如果用户只是闲聊，直接回答即可。"
)

# 5. 测试 Agent
if __name__ == "__main__":
    print("🤖 财报 Agent 已启动！输入 '退出' 结束对话。\n")

    while True:
        user_input = input("你：")
        if user_input in ["退出", "exit", "quit"]:
            print("👋 再见！")
            break

        # 使用 LangGraph 的标准调用方式
        response = agent_executor.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        # 提取最后一条 AI 回复的内容
        ai_message = response["messages"][-1]
        print(f"Agent：{ai_message.content}\n")