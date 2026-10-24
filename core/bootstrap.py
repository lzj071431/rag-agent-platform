"""进程级环境变量引导。

必须在 **任何** 第三方库（transformers / huggingface_hub / chromadb / crewai）
被导入之前执行，否则这些库会在导入阶段就尝试联网。

历史上这些开关散落在 ``config.py``、``crewai_analysis.py`` 等多个文件里，
靠"文件导入顺序"来保证生效，非常脆弱。现在统一收敛到本模块，
并在 ``core.config`` / ``services.embeddings`` / ``services.agent_service``
的 **第一行** 导入它，确保顺序确定。
"""

from __future__ import annotations

import os

#: 必须强制离线：模型已随镜像分发，联网探测会拖慢启动甚至在无网环境直接失败。
_FORCED_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

#: 遥测类开关：连不上时会层层退避重试，既刷屏又拖慢请求。
_DEFAULT_ENV: dict[str, str] = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "ANONYMIZED_TELEMETRY": "False",  # Chroma 的产品遥测
    "CREWAI_DISABLE_TELEMETRY": "1",  # CrewAI 的 OTEL 遥测
    "OTEL_SDK_DISABLED": "true",
}


def setup_runtime_env() -> None:
    """设置离线 / 遥测开关（幂等，可重复调用）。"""
    for key, value in _FORCED_ENV.items():
        os.environ[key] = value
    for key, value in _DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


# 导入即生效：本模块被 import 的那一刻就完成设置。
setup_runtime_env()
