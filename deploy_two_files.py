#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_two_files.py —— 只把指定文件单独同步到 /root/deploy，然后重建容器。

和 auto_deploy.py 的区别：
  auto_deploy.py 上传整个 deploy/ 目录 + 模型（很重）；
  本脚本只上传 FILES 里列出的文件（默认 weekly_report_agent.py 和 .env），
  上传后逐个做 md5 校验，再远程执行 docker(-)compose up -d --build。

配置全部来自 auto_deploy.py（HOST/PORT/USERNAME/REMOTE_DIR）与 .deploy_pw。

用法（必须用项目自带 .venv 的 python —— paramiko 只装在那里）：
    .venv\\Scripts\\python.exe deploy_two_files.py                 # 上传 + 重建
    .venv\\Scripts\\python.exe deploy_two_files.py --dry-run       # 只上传校验，不重建
    .venv\\Scripts\\python.exe deploy_two_files.py --file app.py   # 换文件（可重复）
"""

from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
import socket
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    import auto_deploy as cfg  # 复用 HOST / PORT / USERNAME / REMOTE_DIR 等配置
except SystemExit as exc:  # auto_deploy 在缺少 paramiko 时会 sys.exit
    sys.exit(str(exc))

import paramiko  # noqa: E402  (auto_deploy 已确认可用)

# 默认要单独同步的文件（相对项目根目录）
FILES = ("weekly_report_agent.py", ".env")
PW_FILE = PROJECT_DIR / ".deploy_pw"

CONNECT_TIMEOUT = 20
CMD_TIMEOUT = 60
BUILD_STALL_TIMEOUT = 1800
REMOTE_TMP_SUFFIX = ".uploading"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"\n❌ {msg}")
    sys.exit(code)


def _to_text(data) -> str:
    return data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else data


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_password() -> tuple[str | None, str]:
    pw = os.environ.get("DEPLOY_PASSWORD")
    if pw and pw.strip():
        return pw.strip(), "环境变量 DEPLOY_PASSWORD"
    if PW_FILE.is_file():
        pw = PW_FILE.read_text(encoding="utf-8-sig").strip()
        if pw:
            return pw, f"密码文件 {PW_FILE.name}"
    return None, ""


def preflight_tcp() -> None:
    """先做一次 TCP 探测，让不可达的失败是秒级且信息明确，而不是等 paramiko 超时。"""
    try:
        with socket.create_connection((cfg.HOST, cfg.PORT), timeout=8):
            return
    except OSError as exc:
        die(
            f"服务器 {cfg.HOST}:{cfg.PORT} 无法建立 TCP 连接（{exc}）。\n"
            "  常见原因：实例已停止 / 欠费停机 / 公网 IP 变了 / 安全组未放行 22。\n"
            "  请在阿里云控制台确认实例运行状态与公网 IP 后再重试。"
        )


def run_capture(ssh, cmd: str) -> tuple[int, str]:
    _, out, _ = ssh.exec_command(cmd, timeout=CMD_TIMEOUT)
    rc = out.channel.recv_exit_status()
    return rc, _to_text(out.read()).strip()


def run_stream(ssh, cmd: str, title: str = "", stall: int = BUILD_STALL_TIMEOUT) -> int:
    if title:
        log(f"\n── {title}")
    log(f"$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True, timeout=stall)
    stdin.close()
    while True:
        raw = stdout.readline()
        if not raw:
            break
        line = _to_text(raw).rstrip()
        if line:
            log(f"   {line}")
    return stdout.channel.recv_exit_status()


def remote_md5(ssh, remote: str) -> str:
    rc, out = run_capture(ssh, f"md5sum {remote} 2>/dev/null | awk '{{print $1}}'")
    return out if rc == 0 else ""


def detect_compose(ssh) -> str:
    rc, _ = run_capture(ssh, "docker compose version >/dev/null 2>&1 && echo ok")
    if rc == 0:
        _, ver = run_capture(ssh, "docker compose version")
        log(f"  ✔ {ver}（compose v2 插件）")
        return "docker compose"
    rc, _ = run_capture(ssh, "docker-compose --version >/dev/null 2>&1 && echo ok")
    if rc == 0:
        _, ver = run_capture(ssh, "docker-compose --version")
        log(f"  ✔ {ver}（compose v1 独立版）")
        return "docker-compose"
    die(
        "服务器上没有 docker compose，也没有 docker-compose。\n"
        "  安装 v2 插件：apt-get update && apt-get install -y docker-compose-plugin"
    )
    return ""  # unreachable


def main() -> int:
    ap = argparse.ArgumentParser(description="单独同步指定文件到服务器并重建容器")
    ap.add_argument("--file", action="append", dest="files", help="要同步的文件（相对项目根），可重复")
    ap.add_argument("--dry-run", action="store_true", help="只上传并校验，不执行远程重建")
    ap.add_argument("--no-verify-md5", action="store_true", help="跳过 md5 校验（只比大小）")
    args = ap.parse_args()

    names = tuple(args.files) if args.files else FILES
    locals_: list[Path] = []
    for name in names:
        p = (PROJECT_DIR / name).resolve()
        if not p.is_file():
            die(f"本地文件不存在: {p}")
        locals_.append(p)

    log("=" * 70)
    log("  单独文件同步 -> " + cfg.REMOTE_DIR)
    log(f"  服务器: {cfg.USERNAME}@{cfg.HOST}:{cfg.PORT}")
    for p in locals_:
        log(f"  本地  : {p.name:<28} {p.stat().st_size:>8} B  md5={md5_file(p)}")
    log("=" * 70)

    password, source = load_password()
    if not password:
        die(f"没找到密码：请设置环境变量 DEPLOY_PASSWORD，或创建 {PW_FILE}")
    log(f"  凭据来源: {source}")

    preflight_tcp()

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            cfg.HOST,
            port=cfg.PORT,
            username=cfg.USERNAME,
            password=password,
            timeout=CONNECT_TIMEOUT,
            banner_timeout=CONNECT_TIMEOUT,
            auth_timeout=CONNECT_TIMEOUT,
            allow_agent=False,
            look_for_keys=False,
        )
    except paramiko.AuthenticationException:
        die("认证失败：.deploy_pw 里的密码不正确")
    except Exception as exc:
        die(f"连接失败: {type(exc).__name__}: {exc}")
    ssh.get_transport().set_keepalive(30)
    log("  ✔ SSH 连接成功")

    sftp = ssh.open_sftp()
    try:
        sftp.stat(cfg.REMOTE_DIR)
    except IOError:
        die(f"远端目录不存在: {cfg.REMOTE_DIR}")

    log(f"\n── 上传 {len(locals_)} 个文件 -> {cfg.REMOTE_DIR}")
    t0 = time.time()
    staged: list[tuple[Path, str, str]] = []  # (local, remote, tmp)
    try:
        for local in locals_:
            remote = posixpath.join(cfg.REMOTE_DIR, local.name)
            tmp = remote + REMOTE_TMP_SUFFIX
            before = remote_md5(ssh, remote)
            sftp.put(str(local), tmp)
            staged.append((local, remote, tmp))
            log(f"  ↑ {local.name:<28} {local.stat().st_size:>8} B（远端旧 md5={before or '不存在'}）")
    except Exception as exc:
        for _, _, tmp in staged:
            try:
                sftp.remove(tmp)
            except IOError:
                pass
        sftp.close()
        ssh.close()
        die(f"上传失败: {type(exc).__name__}: {exc}")
    log(f"  ✔ 上传完成（耗时 {time.time() - t0:.1f}s）")

    log("\n── 校验并落位")
    bad: list[str] = []
    for local, remote, tmp in staged:
        want = md5_file(local)
        got_size = sftp.stat(tmp).st_size
        got = remote_md5(ssh, tmp)
        if got_size != local.stat().st_size or (not args.no_verify_md5 and got != want):
            bad.append(f"{local.name}: 本地 {local.stat().st_size}B/{want[:8]} 远端 {got_size}B/{(got or '?')[:8]}")
            continue
        run_capture(ssh, f"mv -f {tmp} {remote} && chmod 644 {remote}")
        log(f"  ✔ {local.name:<28} md5 一致，已覆盖 {remote}")
    if bad:
        for row in bad:
            log(f"  ✘ {row}")
        sftp.close()
        ssh.close()
        die(f"{len(bad)} 个文件校验不通过，已中止（未执行重建）")
    sftp.close()

    rc, uid = run_capture(ssh, "id -u")
    sudo = "" if uid.strip() == "0" else "sudo "
    if sudo:
        log("  ℹ 当前非 root，将使用 sudo")

    if args.dry_run:
        ssh.close()
        log("\n✅ dry-run 结束：文件已同步，未执行重建")
        return 0

    log("\n── 检查 docker 环境")
    rc, _ = run_capture(ssh, "command -v docker >/dev/null 2>&1 && echo ok")
    if rc != 0:
        die("服务器未安装 docker。请先安装：curl -fsSL https://get.docker.com | sh")
    rc, _ = run_capture(ssh, "docker info >/dev/null 2>&1 && echo ok")
    if rc != 0:
        die("docker 守护进程未运行。请执行：systemctl start docker && systemctl enable docker")
    compose = detect_compose(ssh)

    rc, compose_file = run_capture(
        ssh, f"cd {cfg.REMOTE_DIR} && ls -1 {' '.join(cfg.COMPOSE_CANDIDATES)} 2>/dev/null | head -n1"
    )
    if not compose_file:
        die(f"{cfg.REMOTE_DIR} 下找不到 compose 配置文件，无法 up -d --build", code=2)
    log(f"  ✔ 配置文件: {compose_file}")

    ansi = "" if compose == "docker-compose" else "--ansi never "
    deploy_cmd = (
        f"cd {cfg.REMOTE_DIR} && BUILDKIT_PROGRESS=plain "
        f"{sudo}{compose} {ansi}up -d --build"
    )
    rc = run_stream(ssh, deploy_cmd, title="重建并启动容器")
    if rc != 0:
        ssh.close()
        die(f"重建命令失败（退出码 {rc}），请查看上方输出")
    run_stream(ssh, f"cd {cfg.REMOTE_DIR} && {sudo}{compose} ps", stall=CMD_TIMEOUT)
    run_stream(ssh, f"cd {cfg.REMOTE_DIR} && {sudo}{compose} logs --tail=40", stall=CMD_TIMEOUT)
    ssh.close()

    log("\n" + "=" * 70)
    log("  ✅ 同步 + 重建完成")
    log(f"  访问地址: http://{cfg.HOST}:8501")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n已中断。")
        sys.exit(130)
