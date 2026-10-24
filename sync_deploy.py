#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync_deploy.py —— 把项目根目录同步到 deploy/ 部署暂存目录。

背景
----
`deploy/` 是**纯生成产物**：`auto_deploy.py` / `auto_deploy_full.py` 会把它整个
上传到服务器（`/root/deploy`）并在那边 `docker compose up -d --build`。
过去 `deploy/` 靠手工维护，容易与根目录的 `core/`、`services/`、`utils/`
逐渐漂移 —— 典型事故是本地改了 service 但忘了同步，线上跑的还是旧代码。

本脚本把"同步"变成一条命令，保证 `deploy/` 与根目录**逐字节一致**：

    .venv\\Scripts\\python.exe sync_deploy.py

工作流程
--------
1. 备份 `deploy/.env`（这是唯一需要保留的文件，内含真实密钥）；
2. 清空整个 `deploy/` 目录；
3. 按下方**显式清单**从根目录复制代码 / 配置 / 数据文件；
4. 回写 `.env`；
5. 逐文件计算 SHA256，与根目录比对，打印校验结果。

为什么用"显式白名单"而不是"复制所有 .py"
----------------------------------------
根目录里混着**绝不能上传到服务器**的文件：

* `auto_deploy.py` / `auto_deploy_full.py` / `upload_to_server.py` /
  `deploy_two_files.py` —— 含服务器 IP 与部署逻辑（`.gitignore` 已排除）；
* `.deploy_pw` —— 服务器密码明文；
* `test_agent.py` / `evaluate_retrieval.py` / `verify_retrieval.py` /
  `inspect_chunks.py` / `verify_refactor.py` —— 本地测试脚本，非运行时依赖；
* `*.pdf` —— 原始 PDF 体积大（合计约 5MB），已预处理成 `*_clean.txt`。

白名单能同时保证"不漏同步"和"不误上传"。需要新增文件时，改下方常量即可。

安全设计
--------
* `NEVER_COPY` 是第二道防线：即使有人误把敏感文件加进白名单，
  也会在这里被拦下并报错退出。
* 清空前会校验 `deploy` 路径确实是项目根目录下的子目录，
  防止 `DEPLOY_DIR` 被改错时误删别的目录。

用法
----
    python sync_deploy.py                # 执行同步 + 校验
    python sync_deploy.py --dry-run      # 只打印将要做什么，不动文件
    python sync_deploy.py --check        # 只校验当前是否已同步（CI 可用）
    python sync_deploy.py --quiet        # 精简输出
    python sync_deploy.py --no-verify    # 跳过复制后的哈希校验

退出码
------
    0  成功 / 已同步
    1  执行出错（源文件缺失、权限、路径非法等）
    2  --check 模式下检测到漂移
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

# ======================================================================
# 路径
# ======================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEPLOY_DIR: Path = PROJECT_ROOT / "deploy"

# ======================================================================
# 同步清单（唯一真源 = 项目根目录）
# ======================================================================

#: 需要整体同步的 Python 包目录
SYNC_PACKAGES: tuple[str, ...] = ("core", "services", "utils")

#: 运行时核心模块（app.py 及其直接依赖）
RUNTIME_PY: tuple[str, ...] = (
    "app.py",
    "config.py",
    "loader.py",
    "splitter.py",
    "vector_store.py",
    "rag_chain.py",
    "crewai_analysis.py",
    "weekly_report_agent.py",
)

#: 运维 / 数据准备脚本（构建向量库、下载模型、PDF 预处理）
TOOL_PY: tuple[str, ...] = (
    "build_vectorstore.py",
    "download_model.py",
    "merge_txt.py",
    "preprocess_pdf.py",
)

#: 配置与构建文件
CONFIG_FILES: tuple[str, ...] = (
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    ".env.example",
)

#: 运行时需要的数据文件（build_vectorstore.py 依赖）
DATA_FILES: tuple[str, ...] = (
    "competitor_data.txt",
    "finance_clean.txt",
    "yili_clean.txt",
)

#: 清空 deploy 时需要保留的文件（内含真实密钥，绝不覆盖）
PRESERVE_FILES: tuple[str, ...] = (".env",)

#: 绝对禁止进入 deploy 的文件（安全红线，命中即中止）
NEVER_COPY: frozenset[str] = frozenset(
    {
        ".deploy_pw",              # 服务器密码明文
        "auto_deploy.py",          # 含服务器 IP / 部署逻辑
        "auto_deploy_full.py",
        "upload_to_server.py",
        "deploy_two_files.py",
        "sync_deploy.py",          # 本脚本自身，无需上服务器
        "verify_refactor.py",      # 本地验收测试
        "test_agent.py",
        "evaluate_retrieval.py",
        "verify_retrieval.py",
        "inspect_chunks.py",
        "test_set.json",
    }
)

#: 复制时跳过的目录名（任意层级命中即跳过）
SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__", ".git", ".idea", ".vscode", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".venv", "venv", "env",
        "models", "chroma_db", "logs",
    }
)

#: 复制时跳过的后缀
SKIP_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".pyd"})

# 终端颜色（非 TTY 时自动禁用）
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def ok(msg: str) -> str:
    return _c("✔", "32") + " " + msg


def warn(msg: str) -> str:
    return _c("!", "33") + " " + msg


def fail(msg: str) -> str:
    return _c("✘", "31") + " " + msg


def info(msg: str) -> str:
    return _c("·", "36") + " " + msg


# ======================================================================
# 工具函数
# ======================================================================
def sha256_of(path: Path) -> str:
    """计算文件 SHA256（分块读取，避免大文件占内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> list[tuple[Path, Path]]:
    """构建待复制清单，返回 ``[(源路径, deploy 内相对路径), ...]``。

    只收集**实际存在**的文件；缺失的必需文件会在 :func:`validate_sources`
    中统一报错，而不是静默跳过。
    """
    items: list[tuple[Path, Path]] = []

    # 1) 包目录：递归收集所有文件
    for pkg in SYNC_PACKAGES:
        pkg_dir = PROJECT_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(PROJECT_ROOT)
            if _should_skip(rel):
                continue
            items.append((path, rel))

    # 2) 根目录平铺文件
    for name in (*RUNTIME_PY, *TOOL_PY, *CONFIG_FILES, *DATA_FILES):
        src = PROJECT_ROOT / name
        if src.is_file():
            items.append((src, Path(name)))

    return items


def _should_skip(rel: Path) -> bool:
    """判断相对路径是否应跳过。"""
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    if rel.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def validate_sources() -> list[str]:
    """检查必需文件是否齐全，返回缺失项列表。"""
    required = (
        *RUNTIME_PY,
        *CONFIG_FILES,
        *SYNC_PACKAGES,
    )
    missing: list[str] = []
    for name in required:
        if not (PROJECT_ROOT / name).exists():
            missing.append(name)
    return missing


def check_safety(manifest: list[tuple[Path, Path]]) -> list[str]:
    """安全红线检查：清单中不得出现敏感文件。"""
    violations: list[str] = []
    for src, rel in manifest:
        if src.name in NEVER_COPY or rel.name in NEVER_COPY:
            violations.append(rel.as_posix())
    return violations


def guard_deploy_dir() -> None:
    """确认 deploy 路径安全，避免误删。

    要求：必须是项目根目录的直接子目录，且名字为 ``deploy``。
    """
    if DEPLOY_DIR.parent.resolve() != PROJECT_ROOT:
        raise SystemExit(
            fail(f"deploy 路径非法：{DEPLOY_DIR}（必须位于 {PROJECT_ROOT} 下）")
        )
    if DEPLOY_DIR.name != "deploy":
        raise SystemExit(fail(f"deploy 目录名异常：{DEPLOY_DIR.name}，已中止以防误删"))


# ======================================================================
# 核心步骤
# ======================================================================
def snapshot_preserved() -> dict[str, bytes]:
    """读取需要保留的文件内容（当前只有 .env）。

    优先保留 ``deploy/.env``（可能含服务器专用配置）；
    若不存在则回退到根目录 ``.env``。
    """
    preserved: dict[str, bytes] = {}
    for name in PRESERVE_FILES:
        target = DEPLOY_DIR / name
        source = PROJECT_ROOT / name
        if target.is_file():
            preserved[name] = target.read_bytes()
        elif source.is_file():
            preserved[name] = source.read_bytes()
    return preserved


def clear_deploy(quiet: bool) -> int:
    """清空 deploy 目录，返回删除的条目数。"""
    if not DEPLOY_DIR.exists():
        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        return 0

    removed = 0
    for entry in DEPLOY_DIR.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed += 1

    if not quiet:
        print(info(f"已清空 deploy/（移除 {removed} 个条目）"))
    return removed


def copy_manifest(
    manifest: list[tuple[Path, Path]], quiet: bool
) -> tuple[int, int]:
    """按清单复制文件，返回 ``(复制数, 总字节数)``。"""
    copied = 0
    total_bytes = 0

    for src, rel in manifest:
        dest = DEPLOY_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)  # copy2 保留 mtime，便于比对
        copied += 1
        total_bytes += src.stat().st_size
        if not quiet:
            print(f"  ↑ {rel.as_posix():<34} {src.stat().st_size:>9,} B")

    return copied, total_bytes


def restore_preserved(preserved: dict[str, bytes], quiet: bool) -> None:
    """回写保留文件。"""
    for name, data in preserved.items():
        (DEPLOY_DIR / name).write_bytes(data)
        if not quiet:
            print(info(f"已保留 {name}（{len(data):,} B）"))


def verify_sync(manifest: list[tuple[Path, Path]], quiet: bool) -> list[str]:
    """逐文件比对根目录与 deploy 的哈希，返回不一致项。"""
    mismatches: list[str] = []
    checked = 0

    for src, rel in manifest:
        dest = DEPLOY_DIR / rel
        if not dest.is_file():
            mismatches.append(f"{rel.as_posix()}（deploy 中缺失）")
            continue
        if sha256_of(src) != sha256_of(dest):
            mismatches.append(f"{rel.as_posix()}（内容不一致）")
            continue
        checked += 1

    if not quiet:
        print(info(f"哈希校验：{checked}/{len(manifest)} 个文件一致"))
    return mismatches


def verify_extra(manifest: list[tuple[Path, Path]], quiet: bool) -> list[str]:
    """检查 deploy 中是否存在清单之外的残留文件（防止"该删的没删"）。"""
    expected = {rel.as_posix() for _, rel in manifest}
    expected |= {name for name in PRESERVE_FILES}
    extras: list[str] = []

    for path in DEPLOY_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(DEPLOY_DIR).as_posix()
        if rel not in expected:
            extras.append(rel)

    if extras and not quiet:
        for rel in extras:
            print(warn(f"deploy 中存在清单外文件：{rel}"))
    return extras


# ======================================================================
# 三种运行模式
# ======================================================================
def run_check(manifest: list[tuple[Path, Path]], quiet: bool) -> int:
    """只校验，不修改任何文件。"""
    print(_c("── 校验模式（--check）：不修改任何文件", "1"))

    if not DEPLOY_DIR.is_dir():
        print(fail(f"deploy/ 不存在：{DEPLOY_DIR}"))
        return 2

    mismatches = verify_sync(manifest, quiet)
    extras = verify_extra(manifest, quiet)

    if mismatches or extras:
        print()
        for item in mismatches:
            print(fail(f"漂移：{item}"))
        for item in extras:
            print(fail(f"多余：{item}（根目录已无此文件）"))
        print()

        # 两类问题分开计数，避免出现"0 处差异却判定不同步"的误导性提示
        parts: list[str] = []
        if mismatches:
            parts.append(f"{len(mismatches)} 个文件内容不一致/缺失")
        if extras:
            parts.append(f"{len(extras)} 个文件为多余残留")
        print(fail(f"deploy/ 与根目录不同步：{'，'.join(parts)}"))
        print(info("修复：运行 python sync_deploy.py"))
        return 2

    print(ok("deploy/ 与根目录完全一致"))
    return 0


def run_sync(manifest: list[tuple[Path, Path]], quiet: bool, do_verify: bool) -> int:
    """执行同步。"""
    preserved = snapshot_preserved()
    clear_deploy(quiet)

    copied, total_bytes = copy_manifest(manifest, quiet)
    restore_preserved(preserved, quiet)

    print()
    print(
        ok(
            f"同步完成：{copied} 个文件，{total_bytes / 1024:.1f} KB → {DEPLOY_DIR}"
        )
    )

    if not do_verify:
        return 0

    mismatches = verify_sync(manifest, quiet)
    extras = verify_extra(manifest, quiet)
    if mismatches or extras:
        print()
        for item in mismatches:
            print(fail(f"校验失败：{item}"))
        for item in extras:
            print(fail(f"多余文件：{item}"))
        return 1

    print(ok("校验通过：deploy/ 与根目录逐字节一致"))
    return 0


def run_dry_run(manifest: list[tuple[Path, Path]]) -> int:
    """只打印计划，不触碰文件系统。"""
    print(_c("── 演练模式（--dry-run）：不会修改任何文件", "1"))
    print()

    print(f"目标目录：{DEPLOY_DIR}")
    print(f"源目录  ：{PROJECT_ROOT}")
    print()

    preserved = snapshot_preserved()
    print(f"将保留：{', '.join(preserved) or '（无）'}")
    print()

    if DEPLOY_DIR.is_dir():
        existing = sum(1 for _ in DEPLOY_DIR.rglob("*") if _.is_file())
        print(f"将清空：deploy/ 中现有 {existing} 个文件")
    else:
        print("将创建：deploy/（当前不存在）")
    print()

    print(f"将复制 {len(manifest)} 个文件：")
    current_pkg = None
    for src, rel in manifest:
        pkg = rel.parts[0] if len(rel.parts) > 1 else "（根目录）"
        if pkg != current_pkg:
            current_pkg = pkg
            print(f"\n  [{pkg}]")
        print(f"    {rel.as_posix():<36} {src.stat().st_size:>9,} B")

    print()
    print(info("演练结束。确认无误后去掉 --dry-run 执行实际同步。"))
    return 0


# ======================================================================
# 入口
# ======================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_deploy.py",
        description="把项目根目录的代码/配置同步到 deploy/ 部署暂存目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python sync_deploy.py              # 同步并校验\n"
            "  python sync_deploy.py --dry-run    # 预览将要执行的操作\n"
            "  python sync_deploy.py --check      # 只校验是否已同步\n"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不修改文件")
    parser.add_argument("--check", action="store_true", help="只校验是否已同步（不改文件）")
    parser.add_argument("--quiet", action="store_true", help="精简输出")
    parser.add_argument("--no-verify", action="store_true", help="跳过复制后的哈希校验")
    args = parser.parse_args(argv)

    started = time.perf_counter()

    if not args.quiet:
        print(_c("=" * 66, "1"))
        print(_c("  sync_deploy.py —— 同步项目根目录到 deploy/", "1"))
        print(_c("=" * 66, "1"))

    # ---- 前置校验 ----
    try:
        guard_deploy_dir()
    except SystemExit as exc:
        print(str(exc))
        return 1

    missing = validate_sources()
    if missing:
        print(fail(f"根目录缺少必需文件/目录：{', '.join(missing)}"))
        return 1

    manifest = build_manifest()
    if not manifest:
        print(fail("清单为空，无可同步文件"))
        return 1

    violations = check_safety(manifest)
    if violations:
        print(fail(f"安全拦截：以下文件禁止进入 deploy/：{', '.join(violations)}"))
        print(info("请检查 NEVER_COPY 与同步清单"))
        return 1

    # ---- 分派模式 ----
    try:
        if args.dry_run:
            return run_dry_run(manifest)
        if args.check:
            return run_check(manifest, args.quiet)
        code = run_sync(manifest, args.quiet, do_verify=not args.no_verify)
    except OSError as exc:
        print(fail(f"文件操作失败：{type(exc).__name__}: {exc}"))
        return 1

    if not args.quiet:
        print()
        print(info(f"耗时 {time.perf_counter() - started:.2f}s"))
        print(info("下一步：python auto_deploy_full.py（上传并重建容器）"))

    return code


if __name__ == "__main__":
    raise SystemExit(main())
