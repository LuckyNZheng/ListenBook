"""工具模块"""

from listen_book.utils.embedding_util import generate_hybrid_vectors
from listen_book.utils.milvus_util import (
    ensure_chunks_collection,
    ensure_book_name_collection,
    build_hybrid_search_requests,
    execute_hybrid_search,
    build_book_filter_expr,
)
from listen_book.utils.mongo_history_util import (
    save_chat_message,
    get_recent_messages,
    clear_history,
)
from listen_book.utils.task_util import (
    add_running_task,
    add_done_task,
    update_task_status,
    set_task_result,
    get_task_info,
    get_task_result,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from listen_book.utils.sse_util import (
    create_sse_queue,
    get_sse_queue,
    push_sse_event,
    sse_generator,
    SSEEvent,
)

__all__ = [
    "generate_hybrid_vectors",
    "ensure_chunks_collection",
    "ensure_book_name_collection",
    "build_hybrid_search_requests",
    "execute_hybrid_search",
    "build_book_filter_expr",
    "save_chat_message",
    "get_recent_messages",
    "clear_history",
    "add_running_task",
    "add_done_task",
    "update_task_status",
    "set_task_result",
    "get_task_info",
    "get_task_result",
    "TASK_STATUS_PROCESSING",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "create_sse_queue",
    "get_sse_queue",
    "push_sse_event",
    "sse_generator",
    "SSEEvent",
]