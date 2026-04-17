"""查询重写节点"""

import json
import logging
from typing import Dict, Any, List

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.prompts.query_prompt import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriteNode(BaseNode):
    """查询重写节点 - 在检索前重写查询，利用历史上��文"""

    name = "query_rewrite_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """重写查询以提高检索效果"""
        self.log_step("step1", "查询重写开始")

        query = state.get("original_query", "")
        book_names = state.get("book_names", [])
        history = state.get("history", [])

        if not query:
            self.log_step("step2", "无查询内容，跳过重写")
            return {"rewritten_query": query}

        # 从历史中提取最近提到的书名
        recent_books = self._extract_recent_books(history)

        # 构建历史文本
        history_text = self._build_history_text(history)

        try:
            llm = AIClients.get_llm_client(json_mode=False)
            prompt = QUERY_REWRITE_PROMPT.format(
                query=query,
                book_names=json.dumps(book_names, ensure_ascii=False) if book_names else "无",
                recent_books=json.dumps(recent_books, ensure_ascii=False) if recent_books else "无",
                history=history_text,
            )

            response = llm.invoke(prompt)
            rewritten_query = response.content.strip()

            self.log_step("step2", f"查询重写完成: {rewritten_query[:50]}...")
            return {"rewritten_query": rewritten_query}

        except Exception as e:
            self.logger.warning(f"查询重写失败: {e}，使用原查询")
            return {"rewritten_query": query}

    def _extract_recent_books(self, history: List[Dict]) -> List[str]:
        """从历史对话中提取最近提到的书名"""
        books = set()
        for item in history[-5:]:  # 只看最近5条
            book_names = item.get("book_names", [])
            if isinstance(book_names, list):
                books.update(book_names)
        return list(books)

    def _build_history_text(self, history: List[Dict]) -> str:
        """构建历史对话文本"""
        if not history:
            return "无历史对话"

        lines = []
        for item in history[-3:]:  # 只用最近3条
            role = item.get("role", "")
            text = item.get("text", "")
            if text:
                lines.append(f"{role}: {text}")

        return "\n".join(lines) if lines else "无历史对话"
