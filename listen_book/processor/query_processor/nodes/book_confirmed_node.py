"""书名确认节点"""

import json
import logging
from typing import Dict, Any

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.utils.mongo_history_util import get_recent_messages
from listen_book.prompts.query_prompt import BOOK_RECOGNITION_PROMPT

logger = logging.getLogger(__name__)


class BookConfirmedNode(BaseNode):
    """书名识别与确认节点"""

    name = "book_confirmed_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        self.log_step("step1", "书名识别开始")

        query = state.get("original_query", "")
        session_id = state.get("session_id", "")

        # 检查是否已有识别结果（来自缓存检查阶段）
        pre_identified_books = state.get("book_names", [])

        # 获取历史对话
        history = []
        try:
            history = get_recent_messages(session_id, limit=5)
            self.logger.info(f"获取到的历史对话有: {history[:100]}")
        except Exception as e:
            self.logger.warning(f"获取历史失败: {e}")

        # 如果已有识别结果，直接返回（跳过重复识别）
        if pre_identified_books:
            self.log_step("step2", f"使用预先识别的书名: {pre_identified_books}")
            return {
                "history": history,
                "book_names": pre_identified_books,
                "authors": state.get("authors", []),
                "categories": state.get("categories", []),
                "scenes": state.get("scenes", []),
                "intent": state.get("intent", "qa"),
                "is_explicit_book": state.get("is_explicit_book", False),
                "book_confidence": state.get("book_confidence", 0.0),
            }

        # 无预先识别结果，调用 LLM 识别书名
        result = {
            "history": history,
            "book_names": [],
            "authors": [],
            "categories": [],
            "scenes": [],
            "intent": "qa",  # 默认意图
            "is_explicit_book": False,
            "book_confidence": 0.0,
        }

        try:
            llm = AIClients.get_llm_client(json_mode=True)
            prompt = BOOK_RECOGNITION_PROMPT.format(query=query)
            response = llm.invoke(prompt)

            # 获取原始返回内容
            raw_content = response.content
            self.log_step("step_raw", f"LLM原始返回: {raw_content[:200]}...")

            # 尝试解析 JSON（处理可能的格式问题）
            parsed = self._parse_json_response(raw_content)

            result["book_names"] = parsed.get("book_names", [])
            result["authors"] = parsed.get("authors", [])
            result["categories"] = parsed.get("categories", [])
            result["scenes"] = parsed.get("scenes", [])
            result["intent"] = parsed.get("intent", "qa")
            result["is_explicit_book"] = parsed.get("is_explicit", False)
            result["book_confidence"] = parsed.get("confidence", 0.0)

            self.log_step("step2", f"书名={result['book_names']}, 意图={result['intent']}, 置信度={result['book_confidence']}")

        except Exception as e:
            self.logger.error(f"LLM 识别失败: {e}")
            self.log_step("step2", f"LLM 识别异常: {e}")

        return result

    def _parse_json_response(self, raw_content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON，处理可能的格式问题"""
        import re

        # 1. 直接尝试解析
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass

        # 2. 提取 JSON 对象（可能被其他文本包裹）
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, raw_content, re.DOTALL)
        if matches:
            # 找最长的匹配（通常是完整 JSON）
            for match in sorted(matches, key=len, reverse=True):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # 3. 尝试修复常见问题
        cleaned = raw_content.strip()
        # 移除可能的前缀文本
        if '```json' in cleaned:
            cleaned = re.sub(r'^.*?```json\s*', '', cleaned)
            cleaned = re.sub(r'\s*```.*$', '', cleaned)
        if '```' in cleaned:
            cleaned = re.sub(r'^.*?```\s*', '', cleaned)
            cleaned = re.sub(r'\s*```.*$', '', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 4. 返回空结果
        self.logger.warning(f"无法解析 JSON: {raw_content[:100]}")
        return {}