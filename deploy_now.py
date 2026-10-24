#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy_now.py —— 一键上传 deploy/ 到阿里云并重建容器。

与 auto_deploy_full.py 的区别
-----------------------------
* ``auto_deploy_full.py`` 是"从零装机"：装 docker、拉模型、装依赖，适合新服务器；
* 本脚本是**增量发布**：只上传 ``deploy/`` 代码、重建镜像、重启容器。
  日常改完代码后的发布走这条路径，通常 1~3 分钟。

流程
----
1. 本地预检：``deploy/`` 是否齐全、是否已用 ``sync_deploy.py`` 同步；
2. SSH 连接（密码来自 ``DEPLOY_PASSWORD`` 环境变量或 ``.deploy_pw``）；
3. 备份服务器上的 ``/root/deploy/.env``（避免被覆盖丢失线上密钥）；
4. 上传 ``deploy/`` 全部文件（跳过 ``__pycache__`` / ``.env`` 等）；
5. 远程执行 ``docker compose down`` + ``up -d --build``，实时打印日志；
6. 轮询健康检查 ``/_stcore/health``，直到返回 ok 或超时。

用法
----
    set DEPLOY_PASSWORD=你的密码
    .venv\\Scripts\\python.exe deploy_now.py

    .venv\\Scripts\\python.exe deploy_now.py --dry-run     # 只检查，不连接
    .venv\\Scripts\\python.exe deploy_now.py --no-build    # 只重启，不重建镜像

退出码
------
    0  成功（容器 Running 且健康检查通过）
    1  失败
"""

from __future__ import annotations

import argparse
import posixpath
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = PROJECT_ROOT / "deploy"

# ---- 服务器配置（与 auto_deploy_full.py 保持一致）----
HOST = "8.148.199.18"
PORT = 22
USERNAME = "root"
REMOTE_DIR = "/root/deploy"
PUBLIC_URL = f"http://{HOST}:8501"

# ---- 上传时跳过 ----
SKIP_DIR_NAMES = frozenset(
    {"__pycache__", ".git", ".idea", ".vscode", ".venv", "venv",
     "models", "chroma_db", "logs", ".pytest_cache"}
)
#: .env 不上传覆盖（服务器上已有线上密钥；本地 .env 仅用于开发）
SKIP_FILES = frozenset({".env", ".deploy_pw"})

CONNECT_TIMEOUT = 20
UPLOAD_TIMEOUT = 900
BUILD_TIMEOUT = 2700          # docker build 上限 45 分钟（2 核机器装依赖较慢）
HEALTH_TIMEOUT = 300          # 健康检查轮询上限 5 分钟

_USE_COLOR = sys.stdout.isatty()


def _c(t: str, code: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _USE_COLOR else t


def log(msg: str = "") -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    log(_c("✔ ", "32") + msg)


def warn(msg: str) -> None:
    log(_c("! ", "33") + msg)


def err(msg: str) -> None:
    log(_c("✘ ", "31") + msg)


def title(text: str) -> None:
    log()
    log(_c("─" * 68, "1"))
    log(_c(f"▶ {text}", "1"))
    log(_c("─" * 68, "1"))


# ======================================================================
# 凭据
# ======================================================================
def load_password() -> str:
    """获取 SSH 密码：环境变量优先，其次 .deploy_pw，最后交互输入。"""
    import os

    pw = (os.environ.get("DEPLOY_PASSWORD") or "").strip()
    if pw:
        log("  使用环境变量 DEPLOY_PASSWORD")
        return pw

    pw_file = PROJECT_ROOT / ".deploy_pw"
    if pw_file.is_file():
        pw = pw_file.read_text(encoding="utf-8-sig").strip()
        if pw:
            log("  使用 .deploy_pw 中的密码")
            return pw

    import getpass

    pw = getpass.getpass(f"[{USERNAME}@{HOST}] 服务器密码（不回显）: ").strip()
    if not pw:
        raise SystemExit("未提供密码，已中止")
    return pw


# ======================================================================
# 本地预检
# ======================================================================
def collect_files() -> list[tuple[Path, str]]:
    """收集待上传文件，返回 ``[(本地路径, 远端相对路径), ...]``。"""
    if not DEPLOY_DIR.is_dir():
        raise SystemExit(f"deploy/ 不存在：{DEPLOY_DIR}\n请先运行 python sync_deploy.py")

    items: list[tuple[Path, str]] = []
    for path in sorted(DEPLOY_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DEPLOY_DIR)
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        if rel.name in SKIP_FILES:
            continue
        items.append((path, rel.as_posix()))
    return items


def preflight() -> list[tuple[Path, str]]:
    title("第 1 步 / 5：本地预检")

    required = ("app.py", "Dockerfile", "docker-compose.yml", "requirements.txt",
                "core/config.py", "services/rag_service.py")
    missing = [r for r in required if not (DEPLOY_DIR / r).is_file()]
    if missing:
        raise SystemExit(f"deploy/ 缺少必需文件：{', '.join(missing)}\n请先运行 python sync_deploy.py")

    # 清理本地测试/导入产生的缓存，避免被打进镜像
    purged = _purge_caches()

    items = collect_files()
    total = sum(p.stat().st_size for p, _ in items)
    ok(f"deploy/ 就绪：{len(items)} 个文件，{total / 1024:.1f} KB")
    if purged:
        ok(f"已清理 {purged} 个缓存/日志文件")

    # 提示是否与根目录同步（不阻塞，只提醒）
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import sync_deploy

        manifest = sync_deploy.build_manifest()
        drift = sync_deploy.verify_sync(manifest, quiet=True)
        if drift:
            warn(f"检测到 {len(drift)} 处与根目录不同步，建议先运行 sync_deploy.py")
        else:
            ok("deploy/ 与根目录一致")
    except Exception:  # noqa: BLE001 - 校验失败不影响部署
        warn("跳过同步校验（sync_deploy.py 不可用）")

    return items


def _purge_caches() -> int:
    """删除 deploy/ 中的 __pycache__、*.pyc、logs/，返回清理数量。"""
    import shutil

    removed = 0
    for pycache in DEPLOY_DIR.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)
            removed += 1
    for logs in DEPLOY_DIR.rglob("logs"):
        if logs.is_dir():
            shutil.rmtree(logs, ignore_errors=True)
            removed += 1
    for pattern in ("*.pyc", "*.pyo", "*.log"):
        for f in DEPLOY_DIR.rglob(pattern):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ======================================================================
# SSH / SFTP
# ======================================================================
def connect(password: str):
    try:
        import paramiko
    except ImportError:
        raise SystemExit(
            "缺少 paramiko。请用项目虚拟环境执行：\n"
            r"    .venv\Scripts\python.exe deploy_now.py"
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, port=PORT, username=USERNAME, password=password,
        timeout=CONNECT_TIMEOUT, banner_timeout=CONNECT_TIMEOUT,
        auth_timeout=CONNECT_TIMEOUT, look_for_keys=False, allow_agent=False,
    )
    return client


def run(client, cmd: str, timeout: int = 300) -> tuple[int, str]:
    """执行命令并返回 (退出码, 输出)。"""
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    rc = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "replace") + stderr.read().decode("utf-8", "replace")
    return rc, out.strip()


def run_stream(client, cmd: str, stall: int = BUILD_TIMEOUT) -> int:
    """执行命令并实时打印输出。"""
    log(_c(f"$ {cmd}", "36"))
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=stall)
    stdin.close()
    while True:
        line = stdout.readline()
        if not line:
            break
        text = line.rstrip()
        if text:
            log(f"   {text}")
    return stdout.channel.recv_exit_status()


def ensure_remote_dir(sftp, remote: str) -> None:
    parts = remote.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass


def upload(client, items: list[tuple[Path, str]]) -> None:
    title("第 2 步 / 5：上传文件")
    sftp = client.open_sftp()
    try:
        sftp.get_channel().settimeout(UPLOAD_TIMEOUT)
    except Exception:  # noqa: BLE001
        pass

    started = time.time()
    try:
        ensure_remote_dir(sftp, REMOTE_DIR)
        for local, rel in items:
            remote = posixpath.join(REMOTE_DIR, rel)
            ensure_remote_dir(sftp, posixpath.dirname(remote))
            sftp.put(str(local), remote)
            log(f"   ↑ {rel:<34} {local.stat().st_size:>9,} B")
    finally:
        sftp.close()

    ok(f"上传完成：{len(items)} 个文件，耗时 {time.time() - started:.1f}s")


# ======================================================================
# 远程部署
# ======================================================================
def detect_compose(client) -> str:
    """探测可用的 compose 命令。"""
    rc, _ = run(client, "docker compose version >/dev/null 2>&1 && echo ok")
    if rc == 0:
        return "docker compose"
    rc, _ = run(client, "docker-compose --version >/dev/null 2>&1 && echo ok")
    if rc == 0:
        return "docker-compose"
    raise SystemExit("服务器上没有 docker compose / docker-compose，请先安装")


def build_and_start(client, compose: str, rebuild: bool) -> None:
    title("第 3 步 / 5：重建容器")

    rc, out = run(client, "docker --version 2>&1 | head -1")
    log(f"   {out or 'docker 未安装'}")
    if rc != 0:
        raise SystemExit("服务器未安装 docker")

    # 停止旧容器（失败不阻塞，可能本来就没在跑）
    log()
    run_stream(client, f"cd {REMOTE_DIR} && {compose} down --remove-orphans", stall=600)

    # 启动
    log()
    cmd = f"cd {REMOTE_DIR} && {compose} up -d" + (" --build" if rebuild else "")
    rc = run_stream(client, cmd, stall=BUILD_TIMEOUT)
    if rc != 0:
        raise SystemExit(f"容器启动失败（退出码 {rc}）")
    ok("容器已启动")


def wait_healthy(client) -> bool:
    title("第 4 步 / 5：等待健康检查")
    deadline = time.time() + HEALTH_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        rc, out = run(
            client,
            "docker inspect --format='{{.State.Status}}' rag-qa-app 2>/dev/null",
            timeout=30,
        )
        status = out.strip().strip("'")

        rc2, health = run(
            client,
            "curl -fsS -m 5 http://127.0.0.1:8501/_stcore/health 2>/dev/null",
            timeout=30,
        )

        log(f"   [{attempt}] 容器状态={status or '未知'} | 健康检查={'ok' if rc2 == 0 else '未就绪'}")

        if status == "running" and rc2 == 0:
            ok("容器 Running 且健康检查通过")
            return True
        time.sleep(10)

    warn(f"健康检查超时（{HEALTH_TIMEOUT}s）")
    return False


def show_logs(client) -> None:
    title("容器最近日志（末尾 40 行）")
    _, out = run(client, "docker logs --tail 40 rag-qa-app 2>&1", timeout=60)
    for line in out.splitlines():
        log(f"   {line}")


# ======================================================================
# 主流程
# ======================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="上传 deploy/ 到阿里云并重建容器")
    parser.add_argument("--dry-run", action="store_true", help="只做本地预检，不连接服务器")
    parser.add_argument("--no-build", action="store_true", help="只重启容器，不重建镜像")
    args = parser.parse_args(argv)

    log(_c("=" * 68, "1"))
    log(_c("  deploy_now.py —— 发布到阿里云", "1"))
    log(_c("=" * 68, "1"))

    items = preflight()

    if args.dry_run:
        log()
        ok("预检通过（--dry-run 模式，未连接服务器）")
        log(f"   目标：{USERNAME}@{HOST}:{REMOTE_DIR}")
        log(f"   公网地址：{PUBLIC_URL}")
        return 0

    title("连接服务器")
    password = load_password()
    try:
        client = connect(password)
    except Exception as exc:  # noqa: BLE001
        err(f"SSH 连接失败：{type(exc).__name__}: {exc}")
        log()
        log("  排查建议：")
        log("   1) 确认密码正确：set DEPLOY_PASSWORD=你的密码")
        log("   2) 确认安全组放行 22 端口")
        log("   3) 确认实例运行中")
        return 1
    ok(f"已连接 {USERNAME}@{HOST}")

    try:
        rc, info = run(client, "hostname; docker --version 2>&1 | head -1")
        for line in info.splitlines():
            log(f"   {line}")

        # 备份线上 .env
        rc, _ = run(client, f"test -f {REMOTE_DIR}/.env && echo has_env")
        if "has_env" in _:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run(client, f"cp {REMOTE_DIR}/.env {REMOTE_DIR}/.env.bak_{ts}")
            log(f"   已备份线上 .env -> .env.bak_{ts}")
        else:
            warn("服务器上还没有 .env，将无法注入 API Key")
            log("   请先在服务器 /root/deploy/.env 写入 DEEPSEEK_API_KEY")

        upload(client, items)

        compose = detect_compose(client)
        log(f"   compose 命令：{compose}")

        build_and_start(client, compose, rebuild=not args.no_build)
        healthy = wait_healthy(client)

        if not healthy:
            show_logs(client)
            err("部署未通过健康检查")
            return 1

        title("第 5 步 / 5：部署完成")
        ok(f"公网访问地址：{PUBLIC_URL}")
        log()
        log("   请到网页验证：")
        log("     1) RAG 问答：问「妙可蓝多的行权价格是多少」，应回答 15.83 元")
        log("     2) 竞品分析 Agent：生成报告，确认能跑通")
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
