"""查询流程节点基类"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.task_util import add_running_task, add_done_task, add_node_duration, get_task_info
from listen_book.utils import get_sse_queue, get_task_info as _get_task_info, push_sse_event, SSEEvent

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """节点基类"""

    name: str = "base_node"

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """核心处理逻辑，返回需要更新的字段"""
        pass

    def __call__(self, state: QueryGraphState) -> Dict[str, Any]:
        """LangGraph 调用入口"""
        task_id = state.get("task_id", "")
        is_stream = state.get("is_stream", False)
        add_running_task(task_id, self.name)

        # 流式模式下推送进度事件
        if is_stream:
            _push_progress(task_id)

        start = time.time()
        try:
            result = self.process(state)
            duration = time.time() - start
            add_done_task(task_id, self.name)
            add_node_duration(task_id, self.name, duration)

            # 流式模式下推送进度事件
            if is_stream:
                _push_progress(task_id)

            return result
        except Exception as e:
            self.logger.error(f"{self.name} 处理失败: {e}")
            duration = time.time() - start
            add_done_task(task_id, self.name)
            add_node_duration(task_id, self.name, duration)

            # 流式模式下推送进度事件
            if is_stream:
                _push_progress(task_id)

            raise

    def log_step(self, step: str, msg: str):
        """日志输出"""
        self.logger.info(f"[{self.name}] {step}: {msg}")


def _push_progress(task_id: str) -> None:
    """推送进度事件（仅当 SSE 队列存在时）"""
    try:
        if get_sse_queue(task_id) is not None:
            info = _get_task_info(task_id)
            push_sse_event(task_id, SSEEvent.PROGRESS, {
                "status": info.get("status", "processing"),
                "done_list": info.get("done_list", []),
                "running_list": info.get("running_list", []),
            })
    except Exception as e:
        logger.warning(f"推送进度失败: {e}")