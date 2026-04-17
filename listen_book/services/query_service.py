"""查询服务"""

import uuid
import logging
from typing import List, Dict, Any

from listen_book.processor.query_processor.main_graph import query_app
from listen_book.utils.task_util import (
    update_task_status,
    set_task_result,
    get_task_result,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from listen_book.utils.mongo_history_util import get_recent_messages, clear_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:
    """查询服务"""

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id() -> str:
        return uuid.uuid4().hex[:12]

    def run_query_graph(
        self, session_id: str, task_id: str, query: str, is_stream: bool
    ) -> None:
        """运行查询流程"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream,
        }

        try:
            query_app.invoke(init_state)
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