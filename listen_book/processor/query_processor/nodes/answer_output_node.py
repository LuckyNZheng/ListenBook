"""答案生成节点"""

import json
import logging
from typing import Dict, Any, List

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.utils.mongo_history_util import save_chat_message
from listen_book.utils.task_util import set_task_result
from listen_book.utils.sse_util import push_sse_event, SSEEvent
from listen_book.prompts.query_prompt import ANSWER_GENERATION_PROMPT, QUERY_REWRITE_PROMPT
from listen_book.core.config import get_settings

logger = logging.getLogger(__name__)


class AnswerOutputNode(BaseNode):
    """答案生成节点"""

    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        self.log_step("step1", "答案生成开始")

        settings = get_settings()
        query = state.get("original_query", "")
        book_names = state.get("book_names", [])
        docs = state.get("reranked_docs", [])
        is_stream = state.get("is_stream", False)
        session_id = state.get("session_id", "")
        task_id = state.get("task_id", "")

        self.log_step("step2", f"文档数: {len(docs)}, 流式: {is_stream}")

        if not docs:
            self.log_step("step3", "无参考文档")
            empty_answer = "抱歉，未找到相关书籍信息，请尝试其他问题。"
            set_task_result(task_id, "answer", empty_answer)
            return {"answer": empty_answer, "sources": []}

        try:
            llm = AIClients.get_llm_client(json_mode=False)
        except Exception as e:
            self.logger.error(f"LLM 初始化失败: {e}")
            set_task_result(task_id, "answer", "LLM 初始化失败")
            return {"answer": "LLM 初始化失败", "sources": []}

        # 构建上下文
        context = self._build_context(docs, settings.max_context_chars)
        self.log_step("step3", f"上下文长度: {len(context)}")

        # 重写查询（可选）
        rewritten_query = query
        try:
            history_text = self._build_history_text(state.get("history", []))
            rewrite_prompt = QUERY_REWRITE_PROMPT.format(
                query=query,
                book_names=book_names,
                history=history_text
            )
            rewritten = llm.invoke(rewrite_prompt).content
            rewritten_query = rewritten.strip()
            self.log_step("step4", f"查询重写完成")
        except Exception as e:
            self.logger.warning(f"查询重写失败: {e}")

        # 生成答案
        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            book_names=json.dumps(book_names, ensure_ascii=False),
            context=context
        )

        self.log_step("step5", "生成答案")
        if is_stream:
            answer = self._stream_generate(llm, prompt, task_id)
        else:
            try:
                answer = llm.invoke(prompt).content
                set_task_result(task_id, "answer", answer)
            except Exception as e:
                self.logger.error(f"答案生成失败: {e}")
                answer = f"答案生成失败: {e}"
                set_task_result(task_id, "answer", answer)

        self.log_step("step6", "答案生成完成")

        # 构建来源信息
        sources = []
        for doc in docs[:5]:
            sources.append({
                "book_name": doc.get("book_name", ""),
                "author": doc.get("author", ""),
                "content_type": doc.get("content_type", ""),
                "source_file": doc.get("source_file", ""),
            })

        # 保存对话历史
        try:
            save_chat_message(
                session_id=session_id,
                role="user",
                text=query,
                rewritten_query=rewritten_query,
                book_names=book_names,
            )
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                book_names=book_names,
            )
        except Exception as e:
            self.logger.warning(f"保存历史失败: {e}")

        # 流式结束
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL, {
                "answer": answer,
                "book_names": book_names,
                "sources": sources,
            })

        return {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "sources": sources,
        }

    def _build_context(self, docs: List[Dict], max_chars: int) -> str:
        """构建上下文"""
        context_parts = []
        total_chars = 0

        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            book_name = doc.get("book_name", "")
            content_type = doc.get("content_type", "")

            part = f"【来源{i+1}】书名：{book_name}，类型：{content_type}\n内容：{content}\n"

            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n".join(context_parts)

    def _build_history_text(self, history: List[Dict]) -> str:
        """构建历史对话文本"""
        if not history:
            return "无历史对话"

        lines = []
        for item in history:
            role = item.get("role", "")
            text = item.get("text", "")
            lines.append(f"{role}: {text}")

        return "\n".join(lines)

    def _stream_generate(self, llm, prompt: str, task_id: str) -> str:
        """流式生成答案"""
        answer = ""
        try:
            for chunk in llm.stream(prompt):
                delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                answer += delta
                push_sse_event(task_id, SSEEvent.DELTA, {"delta": delta})
        except Exception as e:
            self.logger.warning(f"流式生成失败: {e}")
            answer = llm.invoke(prompt).content

        return answer