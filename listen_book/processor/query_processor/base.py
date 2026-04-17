"""查询流程节点基类"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.task_util import add_running_task, add_done_task, add_node_duration

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
        add_running_task(task_id, self.name)

        start = time.time()
        try:
            result = self.process(state)
            duration = time.time() - start
            add_done_task(task_id, self.name)
            add_node_duration(task_id, self.name, duration)
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 处理失败: {e}")
            duration = time.time() - start
            add_done_task(task_id, self.name)
            add_node_duration(task_id, self.name, duration)
            raise

    def log_step(self, step: str, msg: str):
        """日志输出"""
        self.logger.info(f"[{self.name}] {step}: {msg}")