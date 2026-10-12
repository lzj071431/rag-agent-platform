import os
import shutil

# 注意导入顺序：langchain_community 必须早于 langchain_huggingface 导入。
# 当前环境里 langchain_huggingface 会引用 langchain_core._api 中一个已被移除的符号
# （surface_langchain_deprecation_warnings），单独导入会直接 ImportError；
# 先导入 langchain_community 会完成该兼容性补丁。
import langchain_community  # noqa: F401
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import EMBEDDING_MODEL, PERSIST_DIR

COLLECTION_NAME = "rag_kb"
_SQLITE = "chroma.sqlite3"


def get_embeddings():
    # 强制指定本地的绝对路径（确保模型就在项目文件夹里）
    model_path = os.path.join(os.path.dirname(__file__), "models")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=model_path,      # 明确告诉它，去这里找
        model_kwargs={"local_files_only": True},  # 彻底断网，只认本地
    )


def _clear_chroma_client_cache():
    """清空 Chroma 的进程级 client 缓存。

    ⚠️ 这是 Linux/容器环境下的必需操作，不加会直接导致重建向量库失败。

    原因：Chroma 以 persist_directory 字符串为 key，在
    ``SharedSystemClient._identifier_to_system`` 里做进程级单例缓存，
    缓存的 System 内部持有已打开到 chroma.sqlite3 的连接。

    当 reset_vectorstore() 用 shutil.rmtree 把整个目录删掉后，缓存里那个
    System 的 SQLite 连接仍指向**已被 unlink 的 inode**。此时再执行
    Chroma.from_documents 写入，SQLite 会检测到数据库文件已被移走/删除，
    抛出 SQLITE_READONLY_DBMOVED（扩展码 1032）：
        "attempt to write a readonly database"

    为什么本地 Windows 开发时看不到：Windows 上只要进程还持有句柄，
    目录就删不掉（rmtree 会被拒绝，见 reset_vectorstore 的说明），
    所以旧连接指向的文件依然存在，不会触发该错误。
    而 Linux 允许 unlink 正在被打开的文件，rmtree 静默成功 ——
    于是这个 bug 只在 Linux/容器里暴露。

    清掉缓存后，下一次创建 client 会针对新建的目录建立全新连接。
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception as exc:  # pragma: no cover - 仅作兜底，不影响主流程
        print(f"[向量库] 清理 Chroma client 缓存失败（可忽略）: {type(exc).__name__}: {exc}")


def reset_vectorstore(persist_dir=PERSIST_DIR):
    """尽力删除整个向量库目录。

    ⚠️ 在运行中的 Streamlit 里**通常会失败**：只要进程里还有存活的 Chroma 客户端，
    Windows 就会锁住 chroma.sqlite3，删目录必然被拒。
    所以本函数只当作"顺手清理垃圾分段文件"的辅助手段，
    真正保证"不重复累加"的是 drop_collection()。

    📌 Linux/容器环境注意：此处删除会**成功**（Linux 允许 unlink 打开中的文件），
    因此必须在删完后调用 _clear_chroma_client_cache()，否则残留的旧连接
    会指向已删除的 inode，导致后续写入报 SQLITE_READONLY_DBMOVED(1032)。
    """
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
        if os.path.isdir(persist_dir):
            for entry in os.listdir(persist_dir):
                path = os.path.join(persist_dir, entry)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                except OSError:
                    pass

    # 关键：清掉指向旧 inode 的 client 缓存（Linux 上必须，详见函数注释）
    _clear_chroma_client_cache()

    os.makedirs(persist_dir, exist_ok=True)
    return not os.path.exists(os.path.join(persist_dir, _SQLITE))


def drop_collection(persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME, embeddings=None):
    """用 Chroma 自身的 API 删除集合 —— 数据库级操作，sqlite 被占用时同样有效。

    📌 这是修复"检索失效"的关键。
    Chroma.from_documents 对【已存在】的 collection 是「追加」而不是覆盖
    （实测：同一集合连写 10 条两次 -> count 变成 20）。collection 名固定为
    'rag_kb'，所以反复点"构建向量库"会不断累加：项目里曾出现 1736 条记录里
    只有 434 个不同块（每块重复 4 份）。重复块会占满 top-k 名额 ——
    k=5 时实际只能拿到 2 个不同块，这就是"检索全是免责声明、答案永远进不了前 5"的元凶。
    """
    if not os.path.exists(os.path.join(persist_dir, _SQLITE)):
        return False
    embeddings = embeddings if embeddings is not None else get_embeddings()
    try:
        store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        store.delete_collection()
        print(f"[向量库] 已删除旧集合 '{collection_name}'")
        return True
    except Exception as e:
        print(f"[向量库] 未找到可删除的旧集合（{type(e).__name__}: {str(e)[:80]}）")
        return False


def build_vectorstore(chunks, persist_dir=PERSIST_DIR, reset=True, collection_name=COLLECTION_NAME):
    """构建向量库。reset=True 时先清空旧数据，保证幂等（重复构建不会累加）。"""
    embeddings = get_embeddings()

    if reset:
        # 1) 先试删整个目录：此时还没打开任何 Chroma 客户端，能一次清干净
        #    （顺便清掉无引用的 HNSW 垃圾分段文件）
        purged = reset_vectorstore(persist_dir)
        # 2) 目录被占用（Streamlit 运行中持有连接）时，改用删集合 —— 数据库级操作
        if not purged:
            drop_collection(persist_dir, collection_name, embeddings)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    vectorstore.persist()

    count = vectorstore._collection.count()
    print(f"[向量库] 切分块 {len(chunks)} 个 -> 实际写入 {count} 条向量 -> {persist_dir}")
    if count != len(chunks):
        raise RuntimeError(
            f"向量库条数({count})与切分块数({len(chunks)})不一致，旧集合未被清空。\n"
            f"请停止 Streamlit 后手动删除 {os.path.abspath(persist_dir)}，"
            f"再重新启动并点击“构建向量库”。"
        )
    return vectorstore


def load_vectorstore(persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME):
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
