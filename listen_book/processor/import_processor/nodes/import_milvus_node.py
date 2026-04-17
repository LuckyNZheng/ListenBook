"""写入 Milvus 节点"""

from typing import Dict, Any, List

from listen_book.processor.import_processor.base import BaseNode
from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.utils.client.storage_clients import StorageClients
from listen_book.utils.milvus_util import ensure_chunks_collection, ensure_book_name_collection
from listen_book.core.config import get_settings


class ImportMilvusNode(BaseNode):
    """写入向量库节点"""

    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "写入 Milvus 开始")

        settings = get_settings()
        client = StorageClients.get_milvus_client()

        chunks_collection = settings.chunks_collection
        book_name_collection = settings.book_name_collection
        dim = settings.embedding_dim

        # 确保集合存在
        ensure_chunks_collection(client, chunks_collection, dim)
        ensure_book_name_collection(client, book_name_collection, dim)

        chunks = state.get("chunks", [])
        dense_vectors = state.get("dense_vectors", [])
        sparse_vectors = state.get("sparse_vectors", [])

        if not chunks or not dense_vectors:
            raise ValueError("缺少切片或向量数据")

        # 构建插入数据（包含补充字段）
        insert_data = []
        for i, chunk in enumerate(chunks):
            row = {
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
                "content": chunk.get("content", ""),
                "book_name": state.get("book_name", ""),
                "author": state.get("author", ""),
                "content_type": state.get("content_type", "书籍简介"),
                "category": state.get("category", ""),
                "audio_duration": state.get("audio_duration", ""),
                "source_file": state.get("source_file", ""),
                # 补充字段
                "publisher": state.get("publisher", ""),
                "publish_year": state.get("publish_year", ""),
                "summary": state.get("summary", ""),
                "target_audience": state.get("target_audience", ""),
                "recommend_reason": state.get("recommend_reason", ""),
            }
            insert_data.append(row)

        # 写入切片集合
        client.insert(collection_name=chunks_collection, data=insert_data)
        self.log_step("step2", f"写入 {len(insert_data)} 条切片到 {chunks_collection}")

        # 写入书名集合（每个书名只写入一条，包含更多信息）
        book_name = state.get("book_name", "")
        if book_name and dense_vectors:
            book_row = {
                "dense_vector": dense_vectors[0],
                "sparse_vector": sparse_vectors[0],
                "book_name": book_name,
                "author": state.get("author", ""),
                "category":state.get("category", ""),
                "target_audience":state.get("target_audience", ""),
                "audio_duration":state.get("audio_duration", "")
            }
            client.insert(collection_name=book_name_collection, data=[book_row])
            self.log_step("step3", f"写入书名: {book_name} 到 {book_name_collection}")

        return state