"""core —— 跨模块的基础设施层。

包含：
* :mod:`core.bootstrap` —— 进程级环境变量引导（离线开关、遥测关闭）
* :mod:`core.config`    —— Pydantic Settings 配置中心
* :mod:`core.logger`    —— 结构化日志
* :mod:`core.exceptions`—— 统一业务异常体系
* :mod:`core.prompts`   —— 提示词模板集中管理

注意：本 ``__init__`` 刻意不导入任何子模块，避免 ``from core.config import settings``
时触发重量级依赖（crewai / transformers）的循环导入。
"""

__all__ = ["bootstrap", "config", "logger", "exceptions", "prompts"]
