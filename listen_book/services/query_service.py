"""查询服务"""

import uuid
import logging
from typing import List, Dict, Any

from listen_book.processor.query_processor.main_graph import query_app
from listen_book.processor.query_processor.nodes.book_confirmed_node import BookConfirmedNode
from listen_book.utils.task_util import (
    update_task_status,
    set_task_result,
    get_task_result,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from listen_book.utils.mongo_history_util import get_recent_messages, clear_history
from listen_book.utils.redis_cache_util import get_cached_answer, set_cached_answer
from listen_book.utils.sse_util import push_sse_event, SSEEvent, get_sse_queue
from listen_book.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:
    """查询服务"""

    # 书名识别节点（用于缓存检查前的快速识别）
    _book_confirmed_node = BookConfirmedNode()

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id() -> str:
        return uuid.uuid4().hex[:12]

    def _quick_identify_books(self, query: str, session_id: str) -> List[str]:
        """快速识别书名（用于缓存键生成）"""
        try:
            # 获取历史对话
            history = []
            try:
                history = get_recent_messages(session_id, limit=5)
            except Exception:
                pass

            # 运行书名识别节点
            state = {
                "original_query": query,
                "session_id": session_id,
                "history": history,
            }
            result = self._book_confirmed_node.process(state)
            return result.get("book_names", [])
        except Exception as e:
            logger.warning(f"快速书名识别失败: {e}")
            return []

    def run_query_graph(
        self, session_id: str, task_id: str, query: str, is_stream: bool
    ) -> None:
        """运行查询流程（带缓存检查）"""
        settings = get_settings()
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 1. 快速识别书名
        book_names = self._quick_identify_books(query, session_id)
        logger.info(f"识别书名: {book_names}")

        # 2. 检查缓存（流式和非流式都支持）
        if settings.cache_enabled:
            cached = get_cached_answer(query, book_names)
            if cached:
                logger.info(f"缓存命中，直接返回")
                cached_answer = cached.get("answer", "")
                set_task_result(task_id, "answer", cached_answer)
                update_task_status(task_id, TASK_STATUS_COMPLETED)

                # 流式模式：推送缓存答案
                if is_stream and get_sse_queue(task_id) is not None:
                    # 模拟流式输出，分块推送
                    chunk_size = 50
                    for i in range(0, len(cached_answer), chunk_size):
                        chunk = cached_answer[i:i + chunk_size]
                        push_sse_event(task_id, SSEEvent.DELTA, {"delta": chunk})
                    # 推送最终事件
                    push_sse_event(task_id, SSEEvent.FINAL, {
                        "answer": cached_answer,
                        "book_names": cached.get("book_names", book_names),
                        "sources": cached.get("sources", []),
                    })
                return

        # 3. 缓存未命中，运行完整流程
        init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "book_names": book_names,  # 使用已识别的书名
            "is_stream": is_stream,
        }

        try:
            result = query_app.invoke(init_state)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            rewritten_query = result.get("rewritten_query", "")

            # 4. 写入缓存（流式和非流式都支持）
            if settings.cache_enabled:
                set_cached_answer(
                    query=query,
                    book_names=book_names,
                    answer=answer,
                    sources=sources,
                    rewritten_query=rewritten_query,
                )

            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"查询流程异常: {e}")
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(e))

    def get_task_result(self, task_id: str) -> str:
        """获取任务结果"""
        return get_task_result(task_id, "answer", "")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取历史对话"""
        records = get_recent_messages(session_id, limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "book_names": r.get("book_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        """清空历史"""
        return clear_history(session_id)