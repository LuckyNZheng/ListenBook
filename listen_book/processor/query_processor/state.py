"""查询流程状态类型定义"""

from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """查询流程图状态"""

    # 任务标识
    session_id: str
    task_id: str
    message_id: str

    # 查询信息
    original_query: str
    rewritten_query: str

    # 书籍识别
    book_names: List[str]
    authors: List[str]
    categories: List[str]
    scenes: List[str]
    intent: str  # recommend/detail/qa
    is_explicit_book: bool
    book_confidence: float

    # 书名验证结果
    validated_books: List[str]  # 对齐后的书名

    # 检索结果（多路）
    dense_chunks: List       # 普通向量检索结果
    hyde_chunks: List        # HyDE 检索结果
    rrf_chunks: List         # RRF 融合后的结果
    reranked_docs: List      # Rerank 后的结果

    # 生成
    prompt: str
    answer: str
    sources: List

    # 控制
    history: List
    is_stream: bool


DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "rewritten_query": "",
    "book_names": [],
    "authors": [],
    "categories": [],
    "scenes": [],
    "intent": "qa",
    "is_explicit_book": False,
    "book_confidence": 0.0,
    "validated_books": [],
    "dense_chunks": [],
    "hyde_chunks": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "sources": [],
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本"""
    return copy.deepcopy(DEFAULT_STATE)