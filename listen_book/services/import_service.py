"""导入服务"""

import uuid
import logging
import asyncio
from typing import Dict, Any

from listen_book.processor.import_processor.main_graph import import_app
from listen_book.utils.task_util import (
    update_task_status,
    set_task_result,
    get_task_result,
    get_task_info,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from listen_book.core.paths import get_temp_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImportService:
    """导入服务"""

    @staticmethod
    def generate_task_id() -> str:
        return uuid.uuid4().hex[:12]

    def run_import_graph(
        self,
        task_id: str,
        file_path: str,
        book_name: str = "",
        author: str = "",
        content_type: str = "书籍简介",
        category: str = "",
        audio_duration: str = "",
        source_url: str = "",
    ) -> None:
        """运行导入流程"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        init_state = {
            "task_id": task_id,
            "file_path": file_path,
            "book_name": book_name,
            "author": author,
            "content_type": content_type,
            "category": category,
            "audio_duration": audio_duration,
            "source_url": source_url,
            "md_content": "",  # 需要在 entry_node 后填充
        }

        try:
            # 如果是 MD 文件，直接读取内容
            if file_path.endswith(".md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    init_state["md_content"] = f.read()

            for event in import_app.stream(init_state):
                for key, value in event.items():
                    logger.info(f"节点 {key} 完成")

            final_state = init_state
            chunk_count = len(final_state.get("chunks", []))
            set_task_result(task_id, "chunk_count", chunk_count)
            update_task_status(task_id, TASK_STATUS_COMPLETED)

        except Exception as e:
            logger.error(f"导入流程异常: {e}")
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(e))

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        info = get_task_info(task_id)
        info["chunk_count"] = get_task_result(task_id, "chunk_count", 0)
        info["error"] = get_task_result(task_id, "error", "")
        return info