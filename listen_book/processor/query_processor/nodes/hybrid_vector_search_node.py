"""混合向量检索节点"""

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


class HybridVectorSearchNode(BaseNode):
    """混合向量检索节点"""

    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """只返回需要更新的字段"""
        self.log_step("step1", "混合向量检索开始")

        settings = get_settings()
        query = state.get("rewritten_query", "") or state.get("original_query", "")
        validated_books = state.get("validated_books", [])
        categories = state.get("categories", [])
        intent = state.get("intent", "qa")

        try:
            self.log_step("step2", "初始化嵌入模型和向量库")
            client = StorageClients.get_milvus_client()
            bge_m3 = AIClients.get_bge_m3_client()

            # 生成查询向量
            self.log_step("step3", "生成查询向量")
            vectors = generate_hybrid_vectors(bge_m3, [query])
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
            self.log_step("step4", "执行向量检索")
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
                        chunks.append({
                            "content": hit.get("entity", {}).get("content", ""),
                            "book_name": hit.get("entity", {}).get("book_name", ""),
                            "author": hit.get("entity", {}).get("author", ""),
                            "content_type": hit.get("entity", {}).get("content_type", ""),
                            "entry_name": hit.get("entity", {}).get("entry_name", ""),
                            "category": hit.get("entity", {}).get("category", ""),
                            "source_file": hit.get("entity", {}).get("source_file", ""),
                            "source_url": hit.get("entity", {}).get("source_url", ""),
                            "score": hit.get("distance", 0.0),
                        })

            self.log_step("step5", f"检索完成，共 {len(chunks)} 个结果")
            return {"dense_chunks": chunks}

        except Exception as e:
            self.logger.error(f"混合向量检索失败: {e}")
            self.log_step("step5", f"检索异常: {e}")
            return {"dense_chunks": []}