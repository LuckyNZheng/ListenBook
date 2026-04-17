"""重排序节点"""

import logging
from typing import Dict, Any, List

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.core.config import get_settings

logger = logging.getLogger(__name__)


class RerankerNode(BaseNode):
    """重排序节点"""

    name = "reranker_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        self.log_step("step1", "重排序开始")

        settings = get_settings()
        query = state.get("rewritten_query", "") or state.get("original_query", "")
        # 使用 RRF 融合后的结果
        chunks = state.get("rrf_chunks", [])

        self.log_step("step2", f"待重排序文档数: {len(chunks)}")

        if not chunks:
            self.log_step("step3", "无文档，返回空结果")
            return {"reranked_docs": []}

        # 获取 Reranker 客户端
        try:
            reranker = AIClients.get_reranker_client()
        except Exception as e:
            self.logger.warning(f"Reranker 初始化失败: {e}, 使用原始排序")
            return {"reranked_docs": chunks[:settings.rerank_max_top_k]}

        # 构建重排序输入
        pairs = [[query, chunk.get("content", "")] for chunk in chunks]

        # 执行重排序
        try:
            self.log_step("step3", "执行重排序")
            scores = reranker.compute_score(pairs, normalize=True)

            # 确保 scores 是列表
            if not isinstance(scores, list):
                scores = [scores]

            # 按分数排序
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                score = scores[i] if i < len(scores) else 0.0
                scored_chunks.append({**chunk, "rerank_score": score})

            scored_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            # 取 top_k
            top_k = min(settings.rerank_max_top_k, len(scored_chunks))
            top_chunks = scored_chunks[:top_k]

            self.log_step("step4", f"重排序完成，取 top {top_k} 个结果")
            return {"reranked_docs": top_chunks}

        except Exception as e:
            self.logger.error(f"重排序失败: {e}")
            self.log_step("step4", f"重排序异常: {e}")
            return {"reranked_docs": chunks[:settings.rerank_max_top_k]}