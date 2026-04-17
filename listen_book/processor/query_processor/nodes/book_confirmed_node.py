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

        # 获取历史对话
        history = []
        try:
            history = get_recent_messages(session_id, limit=5)
        except Exception as e:
            self.logger.warning(f"获取历史失败: {e}")

        # 调用 LLM 识别书名
        result = {
            "history": history,
            "book_names": [],
            "authors": [],
            "categories": [],
            "scenes": [],
            "is_explicit_book": False,
            "book_confidence": 0.0,
        }

        try:
            llm = AIClients.get_llm_client(json_mode=True)
            prompt = BOOK_RECOGNITION_PROMPT.format(query=query)
            response = llm.invoke(prompt)
            parsed = json.loads(response.content)

            result["book_names"] = parsed.get("book_names", [])
            result["authors"] = parsed.get("authors", [])
            result["categories"] = parsed.get("categories", [])
            result["scenes"] = parsed.get("scenes", [])
            result["is_explicit_book"] = parsed.get("is_explicit", False)
            result["book_confidence"] = parsed.get("confidence", 0.0)

            self.log_step("step2", f"书名={result['book_names']}, 置信度={result['book_confidence']}")

        except Exception as e:
            self.logger.error(f"LLM 识别失败: {e}")
            self.log_step("step2", f"LLM 识别异常: {e}")

        return result