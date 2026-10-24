"""全局配置中心（Pydantic Settings）。

取代原先散落在 ``config.py`` 里的 ``os.getenv()`` 调用。

设计要点
--------
1. **唯一入口**：全项目只允许通过 ``from core.config import settings`` 读取配置，
   业务代码中不应再出现 ``os.getenv`` / ``os.environ[...]``。
2. **分层模型**：环境相关（``AppSettings``）与调参相关（嵌套的
   ``RetrievalSettings`` / ``ChunkingSettings``）分离，结构清晰。
3. **强校验**：类型错误、必填项缺失会在进程启动时立刻失败（fail fast），
   而不是等到用户点按钮才炸。
4. **密钥不外泄**：``SecretStr`` 保证 Key 不会出现在日志 / ``repr`` 里。
5. **环境变量名保持不变**：``DEEPSEEK_API_KEY``、``FEISHU_WEBHOOK_URL``
   与现有 ``.env`` 完全兼容，无需改动任何部署配置。
6. **相对路径自动转绝对路径**：避免"必须在某个工作目录下运行"的隐式依赖
   （原 ``crewai_analysis.py`` 就是被这个问题坑过）。

与旧 ``config.py`` 的兼容
-------------------------
旧的 ``from config import PERSIST_DIR, MODEL_NAME`` 依然可用 —— 根目录
``config.py`` 已改为本模块的薄转发层。
"""

from __future__ import annotations

# 必须最先导入：在任何重量级第三方库之前完成离线开关设置。
from core import bootstrap as _bootstrap  # noqa: F401  (导入即生效)

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 项目根目录（core/ 的上一级）。所有相对路径都以此为基准。
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: 默认 .env 路径。嵌套 Settings 必须各自声明，否则读不到 .env（见下方说明）。
_ENV_FILE = str(PROJECT_ROOT / ".env")

#: 应用级环境变量前缀。加前缀是为了避免通用变量名（DEBUG / MODEL_NAME / APP_NAME）
#: 被宿主机环境意外劫持（实测：宿主机存在 DEBUG=true 会直接把 settings.debug 改成 True）。
_ENV_PREFIX = "APP_"

#: 允许回退到「无前缀旧名」的变量白名单。
#:
#: 判据：这些名字足够项目专属（``DEEPSEEK_API_KEY`` / ``FEISHU_WEBHOOK_URL`` /
#: ``PERSIST_DIR`` …），被无关程序占用的概率极低，保留旧名可让既有 ``.env``
#: 零改动迁移。
#:
#: 相反，``DEBUG`` / ``MODEL_NAME`` / ``APP_NAME`` / ``LOG_LEVEL`` 这类**通用名**
#: 极易被宿主机、CI、Docker、其它 Python 包污染，因此**刻意不提供旧名回退**，
#: 只认 ``APP_`` 前缀版本。
_LEGACY_FALLBACK_ALLOWED = frozenset(
    {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "LLM_TEMPERATURE",
        "LLM_TIMEOUT_SECONDS",
        "RETRY_MAX_ATTEMPTS",
        "RETRY_INITIAL_WAIT",
        "RETRY_MAX_WAIT",
        "RETRY_EXPONENTIAL_BASE",
        "PERSIST_DIR",
        "COLLECTION_NAME",
        "EMBEDDING_MODEL",
        "FEISHU_WEBHOOK_URL",
        "WEBHOOK_TIMEOUT_SECONDS",
        "WEBHOOK_MAX_CONTENT",
        "LOG_DIR",
        "LOG_FILE_ENABLED",
        "LOG_MAX_BYTES",
        "LOG_BACKUP_COUNT",
        "REPORT_OUTPUT_PATH",
    }
)


def _alias(field_name: str, *extra: str) -> AliasChoices:
    """构造别名链：``APP_<FIELD>`` 优先，必要时回退到旧的无前缀名字。

    仅当旧名在白名单内时才加入回退，避免通用变量名（``DEBUG`` 等）被劫持。
    """
    names = [f"{_ENV_PREFIX}{field_name.upper()}", *extra]
    legacy = field_name.upper()
    if legacy in _LEGACY_FALLBACK_ALLOWED:
        names.append(legacy)
    return AliasChoices(*names)


class ChunkingSettings(BaseSettings):
    """文本切分参数（财报类长文档调优）。

    ⚠️ 必须显式声明 ``env_file``：嵌套 ``BaseSettings`` 由 ``default_factory``
    实例化时**不会继承**父模型的 ``env_file``（实测父模型传入的 .env 对其无效），
    只有 OS 环境变量能生效。若不声明，``.env`` 里的 ``CHUNK_SIZE`` 会被静默忽略。
    """

    model_config = SettingsConfigDict(
        env_prefix="CHUNK_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # 原值 500/50：一个完整的财务/公告段落常被拦腰切断，
    # "行权价格为每份人民币15.83元" 这类关键句容易被切散，故放大到 800/100。
    size: int = Field(default=800, ge=100, le=8000, description="单个文本块的目标字符数")
    overlap: int = Field(default=100, ge=0, description="相邻文本块的重叠字符数")
    section_marker: str = Field(
        default="=====",
        description="多文档合并时的分节标记，splitter 依赖它做按文件分节",
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkingSettings":
        if self.overlap >= self.size:
            raise ValueError(
                f"CHUNK_OVERLAP({self.overlap}) 必须小于 CHUNK_SIZE({self.size})，"
                "否则切分器会死循环"
            )
        return self


class RetrievalSettings(BaseSettings):
    """检索参数（混合检索 + RRF 融合）。

    ⚠️ 同 :class:`ChunkingSettings`：必须显式声明 ``env_file``。
    """

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 原值 k=5：在只有 279~435 个块的库里召回过窄，
    # 关键段落排在第 10~37 名，永远进不了前 5。
    vector_k: int = Field(default=15, ge=1, description="向量检索召回数")
    bm25_k: int = Field(default=15, ge=1, description="关键词(BM25)检索召回数")
    final_k: int = Field(default=10, ge=1, description="RRF 融合后最终交给大模型的块数")
    rrf_c: int = Field(default=60, ge=1, description="RRF 平滑常数（业界惯例 60）")
    # 财报问答里精确数字/专名靠 BM25 命中，故略加权。
    bm25_weight: float = Field(default=1.5, gt=0, description="BM25 在 RRF 中的权重")
    # 关闭时为纯向量检索（仅用于对比排查）。
    enable_hybrid: bool = Field(default=True, description="是否启用向量+BM25 混合检索")
    # 开启后会额外调用 LLM 生成多个改写 query，提升召回但增加 1 次 API 调用与延迟。
    enable_multi_query: bool = Field(default=False, description="是否启用 MultiQuery 改写")
    # CrewAI 工具中判定"检索到有效内容"的 L2 距离阈值。
    # bge-small-zh-v1.5 + Chroma 默认 L2 距离实测：相关 ≈0.38~1.00，无关 ≥1.16。
    relevance_distance_threshold: float = Field(
        default=1.05,
        gt=0,
        description="低于该 L2 距离才认为是真正相关的内容",
    )


class AppSettings(BaseSettings):
    """应用主配置。

    环境变量命名规则
    ----------------
    * 优先读取带前缀的名字，如 ``APP_MODEL_NAME``；
    * 同时保留旧的无前缀名字（``MODEL_NAME`` / ``DEEPSEEK_API_KEY`` /
      ``FEISHU_WEBHOOK_URL`` 等）作为回退，既有 ``.env`` 无需任何改动。

    为什么需要前缀：``DEBUG``、``MODEL_NAME``、``APP_NAME`` 这类通用变量名
    在宿主机 / CI / Docker 环境中极易被其它程序占用，实测宿主机存在
    ``DEBUG=true`` 会把 ``settings.debug`` 直接改成 ``True``。加前缀后
    这些意外注入会被隔离掉。
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix=_ENV_PREFIX,
        extra="ignore",  # 忽略 .env 中与业务无关的变量（如 PATH）
        case_sensitive=False,
    )

    # ---------------- 大模型（DeepSeek，OpenAI 兼容协议）----------------
    deepseek_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=_alias("deepseek_api_key"),
        description="DeepSeek API Key，缺失时调用大模型会抛出 ConfigurationError",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=_alias("deepseek_base_url"),
        description="DeepSeek OpenAI 兼容接口地址",
    )
    model_name: str = Field(
        default="deepseek-chat",
        validation_alias=_alias("model_name"),
        description="对话模型名",
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        validation_alias=_alias("llm_temperature"),
        description="采样温度",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        validation_alias=_alias("llm_timeout_seconds"),
        description="单次 LLM 请求超时",
    )

    # ---------------- 重试策略（Tenacity）----------------
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias=_alias("retry_max_attempts"),
        description="最多尝试次数",
    )
    retry_initial_wait: float = Field(
        default=2.0,
        ge=0,
        validation_alias=_alias("retry_initial_wait"),
        description="首次退避等待秒数（之后 2s→4s→8s 倍增）",
    )
    retry_max_wait: float = Field(
        default=30.0,
        gt=0,
        validation_alias=_alias("retry_max_wait"),
        description="单次退避等待上限秒数",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        gt=1,
        validation_alias=_alias("retry_exponential_base"),
        description="指数退避的底数",
    )

    # ---------------- 向量库 ----------------
    persist_dir: str = Field(
        default="./chroma_db",
        validation_alias=_alias("persist_dir"),
        description="Chroma 持久化目录",
    )
    collection_name: str = Field(
        default="rag_kb",
        validation_alias=_alias("collection_name"),
        description="Chroma 集合名",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        validation_alias=_alias("embedding_model"),
        description="本地 Embedding 模型名",
    )

    # ---------------- 外部集成 ----------------
    feishu_webhook_url: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=_alias("feishu_webhook_url"),
        description="飞书群机器人 Webhook；为空时发送周报会抛出 ConfigurationError",
    )
    webhook_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias=_alias("webhook_timeout_seconds"),
        description="Webhook 超时",
    )
    webhook_max_content: int = Field(
        default=20_000,
        ge=100,
        validation_alias=_alias("webhook_max_content"),
        description="飞书单条消息字符上限（留余量）",
    )

    # ---------------- 日志 ----------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=_alias("log_level"),
        description="日志级别",
    )
    log_format: Literal["text", "json"] = Field(
        default="text",
        validation_alias=_alias("log_format"),
        description="日志格式：text 便于本地阅读，json 便于线上采集",
    )
    log_dir: str = Field(
        default="./logs",
        validation_alias=_alias("log_dir"),
        description="日志文件目录",
    )
    log_file_enabled: bool = Field(
        default=True,
        validation_alias=_alias("log_file_enabled"),
        description="是否同时写日志文件",
    )
    log_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
        validation_alias=_alias("log_max_bytes"),
        description="单个日志文件上限",
    )
    log_backup_count: int = Field(
        default=5,
        ge=0,
        validation_alias=_alias("log_backup_count"),
        description="日志文件轮转保留个数",
    )

    # ---------------- 应用 ----------------
    app_name: str = Field(
        default="AI 智能应用平台",
        validation_alias=_alias("app_name"),
        description="应用名称",
    )
    debug: bool = Field(
        default=False,
        validation_alias=_alias("debug"),
        description="调试模式",
    )
    report_output_path: str = Field(
        default="./competitor_analysis_report.md",
        validation_alias=_alias("report_output_path"),
        description="竞品分析报告落盘路径",
    )

    # ---------------- 嵌套配置 ----------------
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    # ================= 校验与派生属性 =================

    @field_validator("deepseek_base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("DEEPSEEK_BASE_URL 必须以 http:// 或 https:// 开头")
        return v

    @field_validator("persist_dir", "log_dir", "report_output_path")
    @classmethod
    def _resolve_path(cls, v: str) -> str:
        """相对路径统一转换为基于项目根目录的绝对路径。"""
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return str(p.resolve())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> str:
        """本地模型缓存目录（HF 离线模式强制从这里加载）。"""
        return str(PROJECT_ROOT / "models")

    @property
    def has_llm_credentials(self) -> bool:
        """是否配置了大模型密钥（用于 UI 提前给出友好提示）。"""
        return bool(self.deepseek_api_key.get_secret_value().strip())

    @property
    def has_webhook(self) -> bool:
        """是否配置了飞书 Webhook。"""
        return bool(self.feishu_webhook_url.get_secret_value().strip())

    def require_llm_credentials(self) -> str:
        """返回明文 API Key；未配置时抛出可读错误。

        调用方：``services.llm_factory``。
        """
        from core.exceptions import ConfigurationError

        key = self.deepseek_api_key.get_secret_value().strip()
        if not key:
            raise ConfigurationError(
                "未配置 DEEPSEEK_API_KEY",
                detail=f"请在 {PROJECT_ROOT / '.env'} 中设置 DEEPSEEK_API_KEY=sk-xxx",
            )
        return key

    def require_webhook(self) -> str:
        """返回明文 Webhook 地址；未配置时抛出可读错误。"""
        from core.exceptions import ConfigurationError

        url = self.feishu_webhook_url.get_secret_value().strip()
        if not url:
            raise ConfigurationError(
                "未配置 FEISHU_WEBHOOK_URL",
                detail=f"请在 {PROJECT_ROOT / '.env'} 中设置飞书机器人 Webhook 地址",
            )
        return url


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """获取全局配置单例。

    使用 ``lru_cache`` 而非模块级实例，便于测试时用
    ``get_settings.cache_clear()`` 重置。
    """
    return AppSettings()


#: 模块级快捷入口：业务代码统一使用 ``settings``。
settings: AppSettings = get_settings()


__all__ = [
    "AppSettings",
    "ChunkingSettings",
    "RetrievalSettings",
    "PROJECT_ROOT",
    "get_settings",
    "settings",
]
