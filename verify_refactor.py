"""工程化重构验收测试。

覆盖 4 项改造要求的核心断言，可直接用 venv 运行::

    .venv\\Scripts\\python.exe verify_refactor.py

设计为**离线可跑**：不发起真实大模型调用（可通过 --with-llm 开启）。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = "[PASS]", "[FAIL]"
_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> None:
    _results.append((bool(condition), message))
    print(f"{PASS if condition else FAIL} {message}")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ======================================================================
# 1. 目录结构分层
# ======================================================================
def test_layering() -> None:
    section("1. 目录结构分层（业务逻辑与前端解耦）")
    for pkg in ("core", "utils", "services"):
        check((ROOT / pkg / "__init__.py").is_file(), f"{pkg}/ 包存在且有 __init__.py")

    expected = {
        "core": ["config.py", "logger.py", "exceptions.py", "prompts.py", "bootstrap.py"],
        "utils": ["retry.py", "bm25.py", "text_cleaner.py", "webhook.py"],
        "services": [
            "embeddings.py",
            "llm_factory.py",
            "document_service.py",
            "vector_store_service.py",
            "retrieval_service.py",
            "rag_service.py",
            "report_service.py",
            "agent_service.py",
        ],
    }
    for pkg, files in expected.items():
        for f in files:
            check((ROOT / pkg / f).is_file(), f"{pkg}/{f} 存在")

    # app.py 只做 UI：不得直接 import 第三方重依赖
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    forbidden = ["langchain_openai", "langchain_core", "crewai", "chromadb", "langgraph"]
    for lib in forbidden:
        check(
            f"import {lib}" not in app_src and f"from {lib}" not in app_src,
            f"app.py 未直接依赖 {lib}（业务逻辑已下沉到 service 层）",
        )
    check(
        "from services." in app_src,
        "app.py 通过 services 层调用业务逻辑",
    )


# ======================================================================
# 2. Pydantic 配置管理
# ======================================================================
def test_config() -> None:
    section("2. Pydantic 配置管理与数据校验")
    from pydantic import ValidationError

    from core.config import AppSettings, ChunkingSettings, RetrievalSettings, settings

    check(settings.model_name == "deepseek-chat", f"model_name 读取正确: {settings.model_name}")
    check(settings.has_llm_credentials, "DEEPSEEK_API_KEY 已从 .env 加载")
    check(settings.chunking.size == 800, f"嵌套配置 chunking.size={settings.chunking.size}")
    check(
        settings.retrieval.final_k == 10,
        f"嵌套配置 retrieval.final_k={settings.retrieval.final_k}",
    )
    check(
        Path(settings.persist_dir).is_absolute(),
        f"相对路径已转为绝对路径: {settings.persist_dir}",
    )
    check(
        "sk-" not in repr(settings) and "sk-" not in str(settings),
        "SecretStr 生效：repr/str 不泄露 API Key",
    )

    # fail-fast 校验
    try:
        ChunkingSettings(size=100, overlap=200)
        check(False, "overlap >= size 应被拒绝")
    except ValidationError:
        check(True, "overlap >= size 被 ValidationError 拒绝（fail fast）")

    try:
        RetrievalSettings(vector_k=0)
        check(False, "vector_k=0 应被拒绝")
    except ValidationError:
        check(True, "vector_k=0 被 ValidationError 拒绝（fail fast）")

    # 业务代码中不应再有散落的 os.getenv
    offenders: list[str] = []
    for path in list((ROOT / "core").rglob("*.py")) + list(
        (ROOT / "utils").rglob("*.py")
    ) + list((ROOT / "services").rglob("*.py")) + [ROOT / "app.py"]:
        if path.name == "config.py" and path.parent.name == "core":
            continue  # 配置中心自身允许
        if path.name == "bootstrap.py":
            continue  # 环境引导层允许
        src = path.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "os.getenv(" in stripped or "os.environ[" in stripped:
                offenders.append(f"{path.relative_to(ROOT)}:{i}")
    check(not offenders, f"无散落的 os.getenv/os.environ 调用（违规: {offenders or '无'}）")


# ======================================================================
# 3. Tenacity 重试
# ======================================================================
def test_retry() -> None:
    section("3. Tenacity 指数退避重试")
    import httpx

    from utils.retry import build_retrying, is_retryable, retryable

    check(is_retryable(httpx.ReadTimeout("t")), "网络超时 → 可重试")
    check(is_retryable(httpx.ConnectError("c")), "连接错误 → 可重试")
    check(is_retryable(ConnectionError("x")), "ConnectionError → 可重试")

    req = httpx.Request("GET", "http://x")
    check(
        is_retryable(httpx.HTTPStatusError("e", request=req, response=httpx.Response(503, request=req))),
        "HTTP 503 → 可重试",
    )
    check(
        is_retryable(httpx.HTTPStatusError("e", request=req, response=httpx.Response(429, request=req))),
        "HTTP 429 → 可重试",
    )
    check(
        not is_retryable(httpx.HTTPStatusError("e", request=req, response=httpx.Response(404, request=req))),
        "HTTP 404 → 不重试（调用方问题）",
    )
    check(not is_retryable(ValueError("v")), "业务异常 → 不重试")

    # 验证实际尝试次数与退避递增（拦截 sleep，不真等待）
    waits: list[float] = []
    calls = {"n": 0}
    retrying = build_retrying()
    retrying.sleep = lambda s: waits.append(s)

    def boom():
        calls["n"] += 1
        raise ConnectionError("simulated")

    try:
        retrying(boom)
    except ConnectionError:
        pass

    check(calls["n"] == 3, f"最多重试 3 次（实际尝试 {calls['n']} 次）")
    check(len(waits) == 2, f"3 次尝试之间有 2 次退避等待（实际 {len(waits)} 次）")
    check(
        len(waits) >= 2 and waits[1] > waits[0],
        f"等待时间倍增（{['%.2fs' % w for w in waits]}）",
    )

    # 装饰器形式可用
    @retryable(max_attempts=2)
    def always_ok() -> str:
        return "ok"

    check(always_ok() == "ok", "retryable 装饰器不影响正常返回")


# ======================================================================
# 4. 结构化日志
# ======================================================================
def test_logging() -> None:
    section("4. 结构化日志（替换 print）")
    import logging

    from core.logger import JsonFormatter, get_logger, setup_logging

    setup_logging()
    log = get_logger("verify.test")

    root = logging.getLogger()
    check(len(root.handlers) >= 1, f"root logger 已配置 handler（{len(root.handlers)} 个）")

    # 幂等：重复 setup 不应叠加 handler
    before = len(root.handlers)
    setup_logging()
    check(len(root.handlers) == before, "setup_logging 幂等，不重复添加 handler")

    # 日志文件已生成
    from core.config import settings

    log_file = Path(settings.log_dir) / "app.log"
    log.info("验收测试日志写入检查")
    check(log_file.is_file(), f"日志文件已生成: {log_file}")
    if log_file.is_file():
        content = log_file.read_text(encoding="utf-8", errors="replace")
        check("verify.test" in content, "日志包含模块名（便于线上排障）")
        check("验收测试日志写入检查" in content, "日志内容已落盘")
        # 时间戳格式检查
        import re

        check(
            bool(re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)),
            "日志包含时间戳",
        )

    # JSON 格式可用
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    import json

    parsed = json.loads(JsonFormatter().format(record))
    check(parsed.get("message") == "hello", "JSON 格式化器输出可解析的 JSON")

    # 核心代码无 print()
    offenders: list[str] = []
    for path in (
        list((ROOT / "core").rglob("*.py"))
        + list((ROOT / "utils").rglob("*.py"))
        + list((ROOT / "services").rglob("*.py"))
        + [ROOT / "app.py"]
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    check(not offenders, f"核心代码中无 print() 调用（违规: {offenders or '无'}）")


# ======================================================================
# 5. 向后兼容
# ======================================================================
def test_compat() -> None:
    section("5. 向后兼容（旧脚本不失效）")
    import importlib

    for mod in ("config", "loader", "splitter", "vector_store", "rag_chain",
                "crewai_analysis", "weekly_report_agent"):
        try:
            importlib.import_module(mod)
            check(True, f"旧模块 {mod} 仍可导入")
        except Exception as exc:  # noqa: BLE001
            check(False, f"旧模块 {mod} 导入失败: {type(exc).__name__}: {exc}")

    import config as legacy

    check(legacy.CHUNK_SIZE == 800, f"旧常量 config.CHUNK_SIZE={legacy.CHUNK_SIZE}")
    check(legacy.FINAL_K == 10, f"旧常量 config.FINAL_K={legacy.FINAL_K}")
    check(bool(legacy.PERSIST_DIR), "旧常量 config.PERSIST_DIR 可用")


# ======================================================================
# 6. 端到端（可选）
# ======================================================================
def test_e2e() -> None:
    section("6. 端到端功能验证（真实向量库 + 检索）")
    from services.retrieval_service import build_hybrid_retriever
    from services.vector_store_service import load_vectorstore, vectorstore_exists

    if not vectorstore_exists():
        check(False, "本地向量库不存在，跳过端到端检索测试")
        return

    vs = load_vectorstore()
    count = vs._collection.count()
    check(count > 0, f"向量库可加载（{count} 条向量）")

    retriever = build_hybrid_retriever(vs)
    docs = retriever.invoke("妙可蓝多的行权价格是多少？")
    check(len(docs) == 10, f"混合检索返回 final_k=10 块（实际 {len(docs)}）")
    check(
        any("15.83" in d.page_content for d in docs),
        "答案块（含 15.83）命中检索结果",
    )


# ======================================================================
# 7. 独立代码审查回归项
# ======================================================================
def test_review_regressions() -> None:
    """覆盖独立审查发现并已修复的问题，防止回归。"""
    section("7. 独立审查问题回归测试")

    # ---- HIGH 1: .env 中的嵌套配置必须生效 ----
    import os
    import subprocess

    env_path = ROOT / ".env"
    original = env_path.read_text(encoding="utf-8")
    try:
        env_path.write_text(
            original + "\nCHUNK_SIZE=1234\nCHUNK_OVERLAP=7\nRETRIEVAL_VECTOR_K=99\n",
            encoding="utf-8",
        )
        probe = (
            "from core.config import AppSettings;"
            "s=AppSettings();"
            "print(s.chunking.size, s.chunking.overlap, s.retrieval.vector_k)"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
        ).stdout.strip()
        check(out == "1234 7 99", f".env 中的嵌套配置生效（实际: {out}）")
    finally:
        env_path.write_text(original, encoding="utf-8")

    # ---- HIGH 2: 导入顺序不得冻结 HF 离线标志 ----
    probe = (
        "import os;os.environ.pop('HF_HUB_OFFLINE',None);"
        "import services.document_service;"
        "import huggingface_hub.constants as c;"
        "print(c.HF_HUB_OFFLINE)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
    ).stdout.strip()
    check(out == "True", f"document_service 导入后 HF 离线标志仍为 True（实际: {out}）")

    # ---- HIGH 3: 导入库不得劫持宿主 logging ----
    probe = (
        "import logging;"
        "h=logging.StreamHandler();logging.getLogger().addHandler(h);"
        "logging.getLogger().setLevel(logging.DEBUG);"
        "import utils.retry;"
        "print(h in logging.getLogger().handlers, logging.getLogger().level==logging.DEBUG)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
    ).stdout.strip()
    check(out == "True True", f"导入 utils.retry 不破坏宿主 logging（实际: {out}）")

    # ---- HIGH 4: CrewAI LLM 必须带 timeout 且关闭 SDK 重试 ----
    from services.llm_factory import create_crewai_llm

    llm = create_crewai_llm()
    check(llm.timeout is not None, f"CrewAI LLM 设置 timeout（{llm.timeout}）")
    check(
        getattr(llm, "max_retries", None) == 0,
        f"CrewAI LLM 关闭 SDK 重试（max_retries={getattr(llm, 'max_retries', None)}）",
    )

    # ---- MEDIUM 7: 通用环境变量名不得劫持配置 ----
    probe = (
        "import os;"
        "from core.config import AppSettings;"
        "os.environ['DEBUG']='true';os.environ['MODEL_NAME']='hijacked';"
        "s=AppSettings();"
        "print(s.debug, s.model_name)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
    ).stdout.strip()
    check(
        out == "False deepseek-chat",
        f"通用环境变量 DEBUG/MODEL_NAME 无法劫持配置（实际: {out}）",
    )

    # 旧项目专属变量名仍需兼容
    probe = (
        "import os;"
        "from core.config import AppSettings;"
        "os.environ['DEEPSEEK_API_KEY']='sk-legacy';"
        "s=AppSettings();"
        "print(s.deepseek_api_key.get_secret_value())"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
    ).stdout.strip()
    check(out == "sk-legacy", f"旧变量名 DEEPSEEK_API_KEY 仍兼容（实际: {out}）")

    # ---- MEDIUM 8: BM25 纯标点语料不得除零 ----
    from utils.bm25 import SimpleBM25

    try:
        scores = SimpleBM25(["!!", "??"]).scores("test")
        check(scores == [0.0, 0.0], f"BM25 纯标点语料不除零（{scores}）")
    except ZeroDivisionError:
        check(False, "BM25 纯标点语料仍会 ZeroDivisionError")

    # ---- MEDIUM 9: weekly_report_agent 兼容层节点可调用 ----
    import weekly_report_agent as legacy_report

    for name in ("fetch_tasks", "analyze_tasks", "generate_report", "send_report",
                 "app", "workflow", "_build_card", "FEISHU_WEBHOOK", "MAX_CONTENT"):
        check(hasattr(legacy_report, name), f"兼容层暴露 {name}")
    try:
        result = legacy_report.fetch_tasks({"tasks": []})
        check(isinstance(result, dict) and "tasks" in result,
              "fetch_tasks(state) 可正常调用（不再是未绑定方法）")
    except TypeError as exc:
        check(False, f"fetch_tasks(state) 调用失败: {exc}")

    # ---- MEDIUM 10: 不可重试异常不得误报重试次数 ----
    import io
    import logging as _logging

    from utils.retry import retryable

    buf = io.StringIO()
    handler = _logging.StreamHandler(buf)
    handler.setFormatter(_logging.Formatter("%(message)s"))
    _logging.getLogger().addHandler(handler)

    calls = {"n": 0}

    @retryable()
    def _fail_immediately():
        calls["n"] += 1
        raise ValueError("non-retryable")

    try:
        _fail_immediately()
    except ValueError:
        pass
    finally:
        _logging.getLogger().removeHandler(handler)

    check(calls["n"] == 1, f"不可重试异常只调用 1 次（实际 {calls['n']} 次）")
    logged = buf.getvalue()
    check("尝试 1 次" in logged, "日志如实记录 1 次尝试，不再误报 3 次")
    check("重试耗尽" not in logged, "不可重试异常不标记为「重试耗尽」")

    # ---- LOW 11: 分节前缀不得叠加 ----
    from langchain_core.documents import Document

    from services.document_service import split_documents as _split

    out_chunks = _split([Document(page_content="【来源】旧章节\n正文内容",
                                  metadata={"source": "t.txt"})])
    check(
        out_chunks[0].page_content.count("【来源】") == 1,
        f"已有【来源】前缀时不重复叠加（实际: {out_chunks[0].page_content[:30]!r}）",
    )

    # ---- LOW 13: run() 不得修改实例状态 ----
    import inspect

    from services.report_service import WeeklyReportService

    run_src = inspect.getsource(WeeklyReportService.run)
    check(
        "self._send_to_feishu =" not in run_src,
        "WeeklyReportService.run() 不修改实例状态（无跨次污染）",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="额外跑真实大模型调用")
    parser.add_argument("--with-e2e", action="store_true", help="跑向量库端到端检索")
    args = parser.parse_args()

    test_layering()
    test_config()
    test_retry()
    test_logging()
    test_compat()
    test_review_regressions()
    if args.with_e2e:
        test_e2e()
    if args.with_llm:
        section("7. 真实大模型调用")
        from services.rag_service import RAGService
        from services.vector_store_service import load_vectorstore

        result = RAGService(load_vectorstore()).ask("妙可蓝多的行权价格是多少？")
        check("15.83" in result.answer, f"LLM 回答包含正确答案 15.83")
        print(f"  回答: {result.answer[:150]}")

    total = len(_results)
    passed = sum(1 for ok, _ in _results if ok)
    print(f"\n{'=' * 70}")
    print(f"验收结果: {passed}/{total} 通过")
    print("=" * 70)
    if passed != total:
        print("\n失败项:")
        for ok, msg in _results:
            if not ok:
                print(f"  {FAIL} {msg}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
