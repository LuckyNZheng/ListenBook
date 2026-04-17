"""书名验证节点"""

import logging
from typing import Dict, Any, List, Optional

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.utils.client.storage_clients import StorageClients
from listen_book.utils.milvus_util import search_book_name_collection
from listen_book.core.config import get_settings

logger = logging.getLogger(__name__)


class BookValidationNode(BaseNode):
    """书名验证节点 - 验证识别的书名是否存在于知识库"""

    name = "book_validation_node"

    # 书名验证阈值
    VALIDATION_THRESHOLD = 0.6  # 向量相似度阈值

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """验证书名并对齐知识库中的书名"""
        self.log_step("step1", "书名验证开始")

        book_names = state.get("book_names", [])
        intent = state.get("intent", "qa")
        is_explicit = state.get("is_explicit_book", False)
        confidence = state.get("book_confidence", 0.0)

        if not book_names:
            self.log_step("step2", "无识别书名，跳过验证")
            return {"validated_books": []}

        # 如果置信度高且明确指定书名，直接使用
        if is_explicit and confidence >= 0.8:
            self.log_step("step2", f"高置信度书名，直接使用: {book_names}")
            return {"validated_books": book_names}

        # 进行书名对齐验证
        validated_books = []
        try:
            bge_m3 = AIClients.get_bge_m3_client()
            client = StorageClients.get_milvus_client()
            settings = get_settings()

            for book_name in book_names:
                # 搜索相似书名
                hits = search_book_name_collection(
                    client=client,
                    collection_name=settings.book_name_collection,
                    book_name=book_name,
                    bge_m3=bge_m3,
                    limit=3,
                )

                if hits and hits[0].get("score", 0) > self.VALIDATION_THRESHOLD:
                    # 找到匹配书名，使用知识库中的标准书名
                    aligned_name = hits[0].get("book_name", "")
                    validated_books.append(aligned_name)
                    self.log_step("step3", f"书名对齐: {book_name} -> {aligned_name}")
                else:
                    # 未找到匹配，保留原名（可能书名集合不完整）
                    validated_books.append(book_name)
                    self.log_step("step3", f"书名未找到匹配，保留原名: {book_name}")

        except Exception as e:
            self.logger.warning(f"书名验证失败: {e}，使用识别结果")
            validated_books = book_names

        self.log_step("step4", f"验证完成，有效书名: {validated_books}")
        return {"validated_books": validated_books}
