"""入口节点：文件检查与初始化"""

import os
from typing import Dict, Any

from listen_book.processor.import_processor.base import BaseNode
from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.core.paths import get_temp_data_dir


class EntryNode(BaseNode):
    """文件入口检查节点"""

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "文件检查与初始化")

        file_path = state.get("file_path", "")
        if not file_path:
            raise ValueError("缺少 file_path 参数")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 判断文件类型
        ext = os.path.splitext(file_path)[1].lower()
        is_pdf = ext == ".pdf"
        is_md = ext == ".md"

        if not is_pdf and not is_md:
            raise ValueError(f"不支持文件类型: {ext}")

        # 获取文件名（不含扩展名）
        file_name = os.path.basename(file_path)
        file_title = os.path.splitext(file_name)[0]

        # 设置输出目录
        file_dir = os.path.join(get_temp_data_dir(), file_title)
        os.makedirs(file_dir, exist_ok=True)

        state["is_pdf_enabled"] = is_pdf
        state["is_md_enabled"] = is_md
        state["file_title"] = file_title
        state["file_name"] = file_name
        state["file_dir"] = file_dir

        self.log_step("step2", f"文件类型: {ext}, 标题: {file_title}")
        return state