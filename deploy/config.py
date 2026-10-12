import os

# 强制离线模式（必须在导入其他库之前设置）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# ================= 切分参数（财报类长文档调优） =================
# 原值 500/50：一个完整的财务/公告段落常被拦腰切断，
# "行权价格为每份人民币15.83元" 这类关键句容易被切散，故放大到 800/100。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ================= 检索参数 =================
# 原值 k=5：在只有 279~435 个块的库里召回过窄，关键段落排在第 10~37 名，永远进不了前 5。
VECTOR_K = 15      # 向量检索召回数
BM25_K = 15        # 关键词(BM25)检索召回数
FINAL_K = 10       # RRF 融合后最终交给大模型的块数
RRF_C = 60         # RRF 平滑常数（业界惯例 60）
BM25_WEIGHT = 1.5  # BM25 在 RRF 中的权重：财报问答里精确数字/专名靠它命中，故略加权

# 关闭时为纯向量检索（仅用于对比排查）
ENABLE_HYBRID = True
# 开启后会额外调用 LLM 生成多个改写 query，提升召回但增加 1 次 API 调用与延迟
ENABLE_MULTI_QUERY = False

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
PERSIST_DIR = "./chroma_db"

# 多文档合并时的分节标记，splitter.py 依赖它来做"按公司/文件分节 + 上下文前缀"
SECTION_MARKER = "====="
