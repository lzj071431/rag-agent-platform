#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 RAG 项目实际使用的 Embedding 模型：BAAI/bge-small-zh-v1.5

下载到项目下的 models/ 目录，生成 HuggingFace 缓存布局：
    models/models--BAAI--bge-small-zh-v1.5/snapshots/<revision>/...

该布局与 vector_store.py 中的读取方式严格一致：
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                          cache_folder=<项目>/models,
                          model_kwargs={"local_files_only": True})
所以下载完成后即可完全离线运行。

用法：
    python download_model.py            # 通常无需参数
环境变量：
    HF_ENDPOINT  下载源，默认 https://hf-mirror.com（国内直连 huggingface.co 易超时）
"""

import os
from pathlib import Path

REPO_ID = "BAAI/bge-small-zh-v1.5"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# 国内直连 huggingface.co 常超时，默认走镜像；用户已显式设置则不覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402  (需在设置 HF_ENDPOINT 之后导入)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 Embedding 模型: {REPO_ID}")
    print(f"  下载源  : {os.environ['HF_ENDPOINT']}")
    print(f"  保存目录: {MODELS_DIR}")

    snapshot_path = snapshot_download(repo_id=REPO_ID, cache_dir=str(MODELS_DIR))

    # 自检：确认生成的目录正是 vector_store.py 会去查找的位置
    expected = MODELS_DIR / f"models--{REPO_ID.replace('/', '--')}"
    ok = Path(snapshot_path).is_dir() and expected.is_dir()
    total = sum(f.stat().st_size for f in expected.rglob("*") if f.is_file())
    print(f"\n下载完成！模型已保存到：{snapshot_path}")
    print(f"  缓存目录: {expected}")
    print(f"  文件大小: {total / 1024 / 1024:.2f} MB")
    if ok:
        print("  ✔ 自检通过：目录布局与 vector_store.py 的读取路径一致，可离线运行")
        return 0
    print("  ✘ 自检失败：未找到预期的 models--BAAI--bge-small-zh-v1.5 目录")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
