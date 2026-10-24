"""向量库服务（Chroma）。

封装构建 / 加载 / 重置逻辑，并**完整保留**原实现中两个关键踩坑修复：

1. ``reset_vectorstore`` 后必须清 Chroma 的进程级 client 缓存，
   否则 Linux 容器里旧连接指向已 unlink 的 inode，写入报
   ``SQLITE_READONLY_DBMOVED(1032)``。
2. ``Chroma.from_documents`` 对已存在的 collection 是**追加**而非覆盖，
   反复点"构建向量库"会导致重复块占满 top-k。因此 reset 时优先删集合。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import langchain_community  # noqa: F401  (必须先于 langchain_huggingface 导入)
from langchain_community.vectorstores import Chroma

from core.config import settings
from core.exceptions import VectorStoreError
from core.logger import get_logger
from services.embeddings import get_embeddings

logger = get_logger(__name__)

_SQLITE_FILE = "chroma.sqlite3"


def _clear_chroma_client_cache() -> None:
    """清空 Chroma 的进程级 client 缓存。

    ⚠️ 这是 Linux/容器环境下的必需操作，不加会直接导致重建向量库失败。

    原因：Chroma 以 ``persist_directory`` 字符串为 key，在
    ``SharedSystemClient._identifier_to_system`` 里做进程级单例缓存，
    缓存的 System 内部持有已打开到 ``chroma.sqlite3`` 的连接。

    当 :func:`reset_vectorstore` 用 ``shutil.rmtree`` 把整个目录删掉后，
    缓存里那个 System 的 SQLite 连接仍指向**已被 unlink 的 inode**。
    此时再写入，SQLite 会检测到数据库文件已被移走，抛出
    ``SQLITE_READONLY_DBMOVED``（扩展码 1032）："attempt to write a readonly database"。

    为什么本地 Windows 开发时看不到：Windows 上只要进程还持有句柄，目录就删不掉，
    所以旧连接指向的文件依然存在，不会触发该错误。
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        logger.debug("已清理 Chroma client 缓存")
    except Exception as exc:  # noqa: BLE001 - 仅作兜底，不影响主流程
        logger.warning(
            "清理 Chroma client 缓存失败（可忽略）| %s: %s", type(exc).__name__, exc
        )


def reset_vectorstore(persist_dir: str | None = None) -> bool:
    """尽力删除整个向量库目录。

    ⚠️ 在运行中的 Streamlit 里**通常会失败**：只要进程里还有存活的 Chroma 客户端，
    Windows 就会锁住 ``chroma.sqlite3``，删目录必然被拒。
    所以本函数只当作"顺手清理垃圾分段文件"的辅助手段，
    真正保证"不重复累加"的是 :func:`drop_collection`。

    Args:
        persist_dir: 向量库目录；默认取配置。

    Returns:
        ``True`` 表示目录已被彻底清空（无残留 sqlite 文件）。
    """
    target = Path(persist_dir or settings.persist_dir)

    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        if target.is_dir():  # Windows 上可能被占用，退化为逐个删除
            for entry in target.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink()
                except OSError as exc:
                    # 删除失败通常意味着文件仍被占用（Windows 下 Chroma 连接未释放），
                    # 记录下来而不是静默吞掉，否则后续写入报错时无法定位原因。
                    logger.warning(
                        "向量库文件删除失败（可能仍被占用）| path=%s | %s: %s",
                        entry,
                        type(exc).__name__,
                        exc,
                    )

    # 关键：清掉指向旧 inode 的 client 缓存（Linux 上必须）
    _clear_chroma_client_cache()

    target.mkdir(parents=True, exist_ok=True)
    cleared = not (target / _SQLITE_FILE).exists()
    logger.debug("向量库目录已重置 | dir=%s | 彻底清空=%s", target, cleared)
    return cleared


def drop_collection(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> bool:
    """用 Chroma 自身的 API 删除集合 —— 数据库级操作，sqlite 被占用时同样有效。

    📌 这是修复"检索失效"的关键。
    ``Chroma.from_documents`` 对【已存在】的 collection 是「追加」而不是覆盖
    （实测：同一集合连写 10 条两次 → count 变成 20）。collection 名固定为
    ``rag_kb``，所以反复点"构建向量库"会不断累加：项目里曾出现 1736 条记录里
    只有 434 个不同块（每块重复 4 份）。重复块会占满 top-k 名额 ——
    k=5 时实际只能拿到 2 个不同块，这就是"检索全是免责声明、答案永远进不了前 5"的元凶。

    Returns:
        ``True`` 表示成功删除了一个已存在的集合。
    """
    target = Path(persist_dir or settings.persist_dir)
    name = collection_name or settings.collection_name

    if not (target / _SQLITE_FILE).exists():
        return False

    try:
        store = Chroma(
            persist_directory=str(target),
            embedding_function=get_embeddings(),
            collection_name=name,
        )
        store.delete_collection()
        logger.info("已删除旧集合 | collection=%s", name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "未找到可删除的旧集合 | collection=%s | %s: %s",
            name,
            type(exc).__name__,
            str(exc)[:120],
        )
        return False


def build_vectorstore(
    chunks: list,
    persist_dir: str | None = None,
    *,
    reset: bool = True,
    collection_name: str | None = None,
) -> Chroma:
    """构建向量库。

    Args:
        chunks: 文本块列表（LangChain ``Document``）。
        persist_dir: 持久化目录。
        reset: ``True`` 时先清空旧数据，保证幂等（重复构建不会累加）。
        collection_name: 集合名。

    Returns:
        构建完成的 ``Chroma`` 实例。

    Raises:
        VectorStoreError: 写入条数与输入块数不一致（说明旧集合未被清空）。
    """
    target = Path(persist_dir or settings.persist_dir)
    name = collection_name or settings.collection_name

    if not chunks:
        raise VectorStoreError("没有可写入的文本块", detail="chunks 为空")

    embeddings = get_embeddings()

    if reset:
        # 1) 先试删整个目录：此时还没打开任何 Chroma 客户端，能一次清干净
        purged = reset_vectorstore(str(target))
        # 2) 目录被占用（Streamlit 运行中持有连接）时，改用删集合 —— 数据库级操作
        if not purged:
            logger.info("向量库目录被占用，改用删除集合的方式重置")
            drop_collection(str(target), name)

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(target),
            collection_name=name,
        )
        vectorstore.persist()
    except Exception as exc:  # noqa: BLE001
        logger.exception("向量库写入失败 | collection=%s", name)
        raise VectorStoreError(
            "向量库写入失败", detail=f"collection={name}", original=exc
        ) from exc

    count = vectorstore._collection.count()
    logger.info(
        "向量库构建完成 | 切分块=%d -> 实际写入=%d 条 | dir=%s | collection=%s",
        len(chunks),
        count,
        target,
        name,
    )

    if count != len(chunks):
        raise VectorStoreError(
            "向量库条数与切分块数不一致，旧集合未被清空",
            detail=(
                f"实际={count}, 期望={len(chunks)}；"
                f"请停止 Streamlit 后手动删除 {target.resolve()}，再重新构建"
            ),
        )
    return vectorstore


def load_vectorstore(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """加载已存在的向量库。

    Raises:
        VectorStoreError: 加载失败（目录不存在 / 集合损坏）。
    """
    target = Path(persist_dir or settings.persist_dir)
    name = collection_name or settings.collection_name

    try:
        return Chroma(
            persist_directory=str(target),
            embedding_function=get_embeddings(),
            collection_name=name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("向量库加载失败 | dir=%s | collection=%s", target, name)
        raise VectorStoreError(
            "向量库加载失败", detail=f"dir={target}", original=exc
        ) from exc


def vectorstore_exists(persist_dir: str | None = None) -> bool:
    """判断本地是否已有可用的向量库（供 UI 决定是否显示问答入口）。"""
    target = Path(persist_dir or settings.persist_dir)
    if not target.is_dir():
        return False
    if not (target / _SQLITE_FILE).exists():
        return False
    try:
        return any(target.iterdir())
    except OSError as exc:
        logger.warning(
            "检查向量库目录失败 | dir=%s | %s: %s", target, type(exc).__name__, exc
        )
        return False


def delete_collection_of(store: Chroma) -> bool:
    """删除某个已打开实例的集合（UI 重建知识库前调用，释放重复数据）。"""
    try:
        store.delete_collection()
        logger.info("已删除旧集合（释放重复数据）")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("旧集合删除跳过 | %s", type(exc).__name__)
        return False


__all__ = [
    "build_vectorstore",
    "load_vectorstore",
    "reset_vectorstore",
    "drop_collection",
    "vectorstore_exists",
    "delete_collection_of",
]
