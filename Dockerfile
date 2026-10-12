FROM python:3.13-slim

# 设置时区（防止日志时间错乱）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 【重要 1】基础镜像必须是 3.13：
#   requirements.txt 是从本地 Python 3.13.9 的虚拟环境 pip freeze 出来的。
#   原先用 3.10 时，pip 会因版本不兼容而找不到部分 pin（实测报错：
#   Could not find a version that satisfies the requirement networkx==3.6.1，
#   3.10 下最高只能看到 3.4.2，因为 3.5+ 要求 Python >= 3.11）。
#   这类问题会逐个 pin 反复出现，因此直接对齐解释器版本，与冻结环境一致。

# 【重要 2】此处不再安装任何系统依赖（build-essential / libgl1-mesa-glx / libglib2.0-0 已移除）：
#   1) Debian bullseye 已停止常规支持（LTS 于 2026-08 终止），其安全仓库 bullseye-security
#      已从各镜像站下架。实测不论 deb.debian.org 还是清华/阿里镜像，bullseye-security 的
#      .deb 一律 404，apt 步骤必然失败（换镜像源无法解决）。
#   2) 本项目依赖（torch / transformers / pymupdf / chromadb / onnxruntime / tokenizers 等）
#      均有 manylinux 预编译 wheel，无需本地编译，也不需要 libGL。
#   3) 移除后构建不再依赖任何已下线的系统仓库，速度快且稳定。

# 先复制依赖清单（单独一层，代码变动时不会让依赖层缓存失效）
COPY requirements.txt .

# ============================================================
# 【重要 3】torch 必须先装 CPU 版，且必须排在 requirements 之前！
#           这两条 RUN 的先后顺序是整个 Dockerfile 最关键的细节。
#
# 根因（实测踩坑，勿改顺序）：
#   sentence-transformers 无条件依赖 torch>=2.2（其余包对 torch 的依赖
#   都挂在 extra 里，不装 extra 就不会触发）。
#   若先执行 `pip install -r requirements.txt`，pip 解析该依赖时会从
#   清华源下载 torch —— 而 PyPI 上的 linux torch 默认是 **CUDA 版**，
#   会连带拉入 nvidia-cublas / cudnn / cufft ... 约 2.5GB。
#   实测后果：构建跑满 30 分钟仍未完成，磁盘从 8.5G 涨到 11G+，
#   镜像最终膨胀到 10.9GB（阿里云 ECS 无 GPU，这些包纯属浪费）。
#
# 正确做法：先把 CPU 版 torch 装好，requirements 阶段 pip 发现
#   torch>=2.2 已被 2.14.0+cpu 满足，就不会再去下载 CUDA 版。
# ============================================================
RUN pip install --no-cache-dir torch==2.14.0 \
      --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖（此时 torch 已就绪，不会再被解析到 CUDA 版本）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 自检：确认装进来的确实是 CPU 版，而不是 CUDA 版。
# 若此处失败，说明顺序被改动或索引源解析异常，应立刻中断构建而不是
# 产出一个 10.9GB 的镜像才发现问题。
RUN python -c "import torch; v=torch.__version__; \
      assert '+cpu' in v, 'torch 不是 CPU 版: ' + v; \
      print('torch 版本校验通过:', v)"

# 复制项目代码（放最后：代码改动只让这一层缓存失效）
COPY . .

EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
