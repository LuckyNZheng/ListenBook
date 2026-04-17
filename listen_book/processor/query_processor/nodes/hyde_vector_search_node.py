"""HyDE 假设性问题检索节点"""

import json
import logging
from typing import Dict, Any, List

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.utils.client.storage_clients import StorageClients
from listen_book.utils.embedding_util import generate_hybrid_vectors
from listen_book.utils.milvus_util import (
    build_hybrid_search_requests,
    execute_hybrid_search,
    build_combined_filter_expr,
)
from listen_book.core.config import get_settings

logger = logging.getLogger(__name__)


HYDE_PROMPT = """你是一位书籍知识库专家。请针对用户的问题，生成一个假设性的详细答案。

用户问题：{query}
相关书籍：{book_names}

要求：
1. 假设知识库中有完美匹配的内容，生成一个详细、准确的答案
2. 答案应该包含书籍推荐理由、书籍详情、作者信息等
3. 答案风格要自然流畅，适合听书场景
4. 不要标注"假设"、"示例"等字样，直接生成答案

请直接生成假设性答案："""


class HydeVectorSearchNode(BaseNode):
    """HyDE 假设性问题检索节点"""

    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """只返回需要更新的字段"""
        self.log_step("step1", "HyDE 检索开始")

        settings = get_settings()
        query = state.get("rewritten_query", "") or state.get("original_query", "")
        validated_books = state.get("validated_books", [])
        categories = state.get("categories", [])
        intent = state.get("intent", "qa")

        if not query:
            self.log_step("step2", "无查询内容，返回空结果")
            return {"hyde_chunks": []}

        # 生成假设性答案
        self.log_step("step2", "调用 LLM 生成假设答案")
        try:
            hyde_answer = self._generate_hyde_answer(query, book_names)
            if not hyde_answer:
                self.log_step("step3", "LLM 未生成假设答案，返回空结果")
                return {"hyde_chunks": []}
            self.log_step("step3", f"假设答案生成完成，长度: {len(hyde_answer)}")
        except Exception as e:
            self.logger.error(f"生成假设答案失败: {e}")
            self.log_step("step3", f"生成假设答案异常: {e}")
            return {"hyde_chunks": []}

        # 执行检索
        try:
            self.log_step("step4", "初始化嵌入模型和向量库")
            bge_m3 = AIClients.get_bge_m3_client()
            client = StorageClients.get_milvus_client()

            # 嵌入：原始问题 + 假设答案
            hyde_text = f"{query}\n{hyde_answer}"
            self.log_step("step5", "生成向量嵌入")
            vectors = generate_hybrid_vectors(bge_m3, [hyde_text])
            dense_vector = vectors["dense"][0]
            sparse_vector = vectors["sparse"][0]

            # 构建过滤条件（使用验证后的书名和意图）
            expr = build_combined_filter_expr(
                book_names=validated_books if validated_books else None,
                categories=categories if categories else None,
                intent=intent,
            )

            # 构建检索请求
            requests = build_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=settings.hybrid_search_limit,
            )

            # 输出字段
            output_fields = [
                "content", "book_name", "author", "content_type",
                "entry_name", "category", "source_file", "source_url"
            ]

            # 执行混合检索
            self.log_step("step6", "执行向量检索")
            results = execute_hybrid_search(
                client=client,
                collection_name=settings.chunks_collection,
                requests=requests,
                output_fields=output_fields,
                limit=settings.hybrid_search_limit,
            )

            # 解析结果
            chunks = []
            if results:
                for hits in results:
                    for hit in hits:
                        entity = hit.get("entity", {})
                        chunks.append({
                            "content": entity.get("content", ""),
                            "book_name": entity.get("book_name", ""),
                            "author": entity.get("author", ""),
                            "content_type": entity.get("content_type", ""),
                            "entry_name": entity.get("entry_name", ""),
                            "category": entity.get("category", ""),
                            "source_file": entity.get("source_file", ""),
                            "source_url": entity.get("source_url", ""),
                            "score": hit.get("distance", 0.0),
                        })

            self.log_step("step7", f"HyDE 检索完成，共 {len(chunks)} 个结果")
            return {"hyde_chunks": chunks}

        except Exception as e:
            self.logger.error(f"HyDE 检索失败: {e}")
            self.log_step("step7", f"HyDE 检索异常: {e}")
            return {"hyde_chunks": []}

    def _generate_hyde_answer(self, query: str, book_names: List[str]) -> str:
        """生成假设性答案"""
        try:
            llm = AIClients.get_llm_client(json_mode=False)
            prompt = HYDE_PROMPT.format(
                query=query,
                book_names=json.dumps(book_names, ensure_ascii=False) if book_names else "未知"
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.warning(f"生成假设答案失败: {e}")
            raise