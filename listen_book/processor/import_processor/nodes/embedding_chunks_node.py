"""向量生成节点"""

from typing import Dict, Any, List

from listen_book.processor.import_processor.base import BaseNode
from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.utils.embedding_util import generate_hybrid_vectors
from listen_book.core.config import get_settings


class EmbeddingChunksNode(BaseNode):
    """向量生成节点"""

    name = "embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "向量生成开始")

        settings = get_settings()
        batch_size = settings.embedding_batch_size

        chunks = state.get("chunks", [])
        if not chunks:
            raise ValueError("缺少切片数据")

        bge_m3 = AIClients.get_bge_m3_client()

        # 提取内容文本
        contents = [chunk.get("content", "") for chunk in chunks]

        # 分批生成向量
        all_dense = []
        all_sparse = []

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            self.log_step("batch", f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 个")

            vectors = generate_hybrid_vectors(bge_m3, batch)
            all_dense.extend(vectors["dense"])
            all_sparse.extend(vectors["sparse"])

        state["dense_vectors"] = all_dense
        state["sparse_vectors"] = all_sparse

        self.log_step("step2", f"向量生成完成，共 {len(all_dense)} 个")
        return state