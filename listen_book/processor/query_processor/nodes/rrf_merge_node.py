"""RRF 多路检索结果融合节点"""

import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from listen_book.processor.query_processor.base import BaseNode
from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.core.config import get_settings

logger = logging.getLogger(__name__)


class RrfMergeNode(BaseNode):
    """RRF (Reciprocal Rank Fusion) 多路检索融合节点"""

    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        self.log_step("step1", "RRF 融合开始")

        settings = get_settings()
        rrf_k = 60  # RRF 平滑参数

        # 获取两路检索结果
        dense_chunks = state.get("dense_chunks", [])
        hyde_chunks = state.get("hyde_chunks", [])

        self.log_step("step2", f"dense: {len(dense_chunks)}, hyde: {len(hyde_chunks)}")

        # 定义权重（两路权重相等）
        search_results = [
            (self._validate_chunks(dense_chunks), 1.0),
            (self._validate_chunks(hyde_chunks), 1.0),
        ]

        # RRF 融合
        merged = self._merge_rrf(search_results, rrf_k, settings.hybrid_search_limit)

        self.log_step("step3", f"融合完成，共 {len(merged)} 个结果")
        return {"rrf_chunks": merged}

    def _validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """校验并提取有效结果"""
        if not chunks:
            return []
        valid = []
        for chunk in chunks:
            if not chunk or not isinstance(chunk, dict):
                continue
            if not chunk.get("content"):
                continue
            valid.append(chunk)
        return valid

    def _merge_rrf(
        self,
        rrf_inputs: List[Tuple[List[Dict], float]],
        k: int,
        max_results: int
    ) -> List[Dict]:
        """RRF 公式融合多路结果

        公式：score(doc) = sum(weight / (k + rank(doc)))
        """
        # 用 content 作为唯一标识（实际应该用 chunk_id）
        chunk_scores = defaultdict(float)
        chunk_data = {}

        for chunks, weight in rrf_inputs:
            for rank, chunk in enumerate(chunks):
                # 用 content 前 100 字符作为标识
                chunk_key = chunk.get("content", "")[:100]
                if not chunk_key:
                    continue

                # RRF 分数累加
                chunk_scores[chunk_key] += weight / (k + rank + 1)

                # 保存数据（第一次出现时保存）
                if chunk_key not in chunk_data:
                    chunk_data[chunk_key] = chunk

        # 按分数排序
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 返回 top_k
        merged = []
        for chunk_key, score in sorted_results[:max_results]:
            chunk = chunk_data.get(chunk_key)
            if chunk:
                chunk["rrf_score"] = score
                merged.append(chunk)

        return merged