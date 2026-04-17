"""导入流程状态类型定义"""

from typing import TypedDict, List
import copy


class ImportGraphState(TypedDict, total=False):
    """导入流程图状态"""

    # 任务标识
    task_id: str

    # 控制标志
    is_md_enabled: bool
    is_pdf_enabled: bool

    # 路径信息
    file_path: str
    file_dir: str
    md_path: str

    # 文件信息
    file_title: str
    file_name: str

    # 书籍基本信息
    book_name: str
    author: str
    content_type: str
    entry_name: str
    category: str
    audio_duration: str
    source_url: str

    # LLM 补充的信息
    publisher: str
    publish_year: str
    summary: str
    target_audience: str
    recommend_reason: str

    # 处理中间数据
    md_content: str
    chunks: List

    # 向量数据
    dense_vectors: List
    sparse_vectors: List


GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_md_enabled": False,
    "is_pdf_enabled": False,
    "file_path": "",
    "file_dir": "",
    "md_path": "",
    "file_title": "",
    "file_name": "",
    "book_name": "",
    "author": "",
    "content_type": "",
    "entry_name": "",
    "category": "",
    "audio_duration": "",
    "source_url": "",
    "publisher": "",
    "publish_year": "",
    "summary": "",
    "target_audience": "",
    "recommend_reason": "",
    "md_content": "",
    "chunks": [],
    "dense_vectors": [],
    "sparse_vectors": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取默认状态副本"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)