"""BGE-M3 混合向量生成：dense + sparse。"""
from typing import Dict, List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction


def generate_hybrid_vectors(
    model: BGEM3EmbeddingFunction, documents: List[str]
) -> Dict:
    """
    生成混合向量。
    Returns: {"dense": [[...], ...], "sparse": [{token_id: weight}, ...]}
    """
    if not documents:
        raise ValueError("documents 不能为空")
    if not all(isinstance(d, str) and d.strip() for d in documents):
        raise ValueError("documents 存在空字符串或非字符串元素")

    result = model.encode_documents(documents)
    if "dense" not in result or "sparse" not in result:
        raise RuntimeError(f"嵌入结果缺字段: {list(result.keys())}")

    # 解析 CSR 稀疏矩阵 → dict 列表
    csr = result["sparse"]
    sparse_list = []
    for i in range(len(documents)):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        token_ids = csr.indices[start:end].tolist()
        weights = csr.data[start:end].tolist()
        sparse_list.append(dict(zip(token_ids, weights)))

    return {
        "dense": [d.tolist() for d in result["dense"]],
        "sparse": sparse_list,
    }


def extract_sparse_from_csr(csr, index: int) -> Dict:
    """从 CSR 矩阵提取第 index 行作为 {token_id: weight}。"""
    start, end = csr.indptr[index], csr.indptr[index + 1]
    token_ids = csr.indices[start:end].tolist()
    weights = csr.data[start:end].tolist()
    return dict(zip(token_ids, weights))
