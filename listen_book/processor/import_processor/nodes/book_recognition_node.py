"""书名识别节点"""

import json
import re
from typing import Dict, Any

from listen_book.processor.import_processor.base import BaseNode
from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.prompts.import_prompt import BOOK_NAME_EXTRACT_PROMPT


class BookRecognitionNode(BaseNode):
    """书名识别节点"""

    name = "book_recognition_node"

    # 书名常见后缀，需要清理
    BOOK_NAME_SUFFIXES = [
        "简介", "介绍", "摘要", "概述", "读书笔记", "书评",
        "推荐", "评论", "读后感", "精华", "要点", "总结",
        "简介版", "精简版", "缩写版", "有声书", "音频版",
    ]

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "书名识别开始")

        # 如果用户已提供书名，则清理后使用
        user_book_name = state.get("book_name", "")
        if user_book_name:
            cleaned_name = self._clean_book_name(user_book_name)
            state["book_name"] = cleaned_name
            self.log_step("step2", f"用户提供书名: {user_book_name} -> 清理后: {cleaned_name}")
            return state

        file_title = state.get("file_title", "")
        md_content = state.get("md_content", "")

        # 取内容样本（前 2000 字符）
        content_sample = md_content[:2000] if md_content else ""

        if not content_sample:
            self.log_step("step2", "无内容样本，清理文件标题作为书名")
            state["book_name"] = self._clean_book_name(file_title)
            return state

        try:
            llm = AIClients.get_llm_client(json_mode=True)
            prompt = BOOK_NAME_EXTRACT_PROMPT.format(
                file_title=file_title,
                content_sample=content_sample
            )
            response = llm.invoke(prompt)
            result = json.loads(response.content)

            raw_book_name = result.get("book_name", "")
            if raw_book_name:
                state["book_name"] = self._clean_book_name(raw_book_name)
            else:
                state["book_name"] = self._clean_book_name(file_title)

            state["author"] = result.get("author", "")
            state["category"] = result.get("category", "")
            state["audio_duration"] = result.get("audio_duration", "")

            self.log_step("step3", f"识别结果: 书名={state['book_name']}, 作者={state['author']}")
        except Exception as e:
            self.logger.warning(f"LLM 识别失败: {e}, 清理文件标题作为书名")
            state["book_name"] = self._clean_book_name(file_title)

        return state

    def _clean_book_name(self, name: str) -> str:
        """清理书名，去掉常见后缀"""
        if not name:
            return ""

        # 去掉书名号
        name = re.sub(r"^《(.+)》$", r"\1", name)
        name = re.sub(r"^【(.+)】$", r"\1", name)

        # 去掉常见后缀
        for suffix in self.BOOK_NAME_SUFFIXES:
            # 匹配后缀（可能有空格或无空格）
            pattern = rf"\s*{suffix}$"
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)

        # 去掉末尾的空白和标点
        name = name.strip()
        name = re.sub(r"[：:、，,。.！!？?]+$", "", name)

        return name.strip()