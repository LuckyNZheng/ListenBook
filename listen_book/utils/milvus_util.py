"""Milvus 操作工具：集合 schema / 索引 / 混合检索。"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker

logger = logging.getLogger(__name__)


# ---------- 集合创建 ----------
def ensure_chunks_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    """创建听书切片集合（不存在则创建）。"""
    if client.has_collection(collection_name):
        return

    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("book_name", DataType.VARCHAR, max_length=500)
    schema.add_field("author", DataType.VARCHAR, max_length=200)
    schema.add_field("content_type", DataType.VARCHAR, max_length=50)
    schema.add_field("entry_name", DataType.VARCHAR, max_length=500)
    schema.add_field("category", DataType.VARCHAR, max_length=200)
    schema.add_field("audio_duration", DataType.VARCHAR, max_length=50)
    schema.add_field("source_file", DataType.VARCHAR, max_length=500)
    schema.add_field("source_url", DataType.VARCHAR, max_length=1000)
    # 新增补充字段
    schema.add_field("publisher", DataType.VARCHAR, max_length=200)
    schema.add_field("publish_year", DataType.VARCHAR, max_length=20)
    schema.add_field("summary", DataType.VARCHAR, max_length=1000)
    schema.add_field("target_audience", DataType.VARCHAR, max_length=500)
    schema.add_field("recommend_reason", DataType.VARCHAR, max_length=1000)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    logger.info(f"集合 {collection_name} 创建成功")


def ensure_book_name_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    """创建书名集合（用于书名模糊对齐）。"""
    if client.has_collection(collection_name):
        return

    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("book_name", DataType.VARCHAR, max_length=500)
    schema.add_field("author", DataType.VARCHAR, max_length=200)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    logger.info(f"集合 {collection_name} 创建成功")


# ---------- 混合检索 ----------
def build_hybrid_search_requests(
    dense_vector: List[float],
    sparse_vector: Dict,
    expr: Optional[str] = None,
    limit: int = 5,
) -> List[AnnSearchRequest]:
    """构建 dense + sparse 的混合检索请求。"""
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        expr=expr,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        expr=expr,
        limit=limit,
    )
    return [dense_req, sparse_req]


def execute_hybrid_search(
    client: MilvusClient,
    collection_name: str,
    requests: List[AnnSearchRequest],
    output_fields: List[str],
    ranker_weights: Tuple[float, float] = (0.5, 0.5),
    limit: int = 5,
) -> List[List[Dict[str, Any]]]:
    """执行混合检索（稠密 + 稀疏 WeightedRanker 融合）。"""
    ranker = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=True)
    return client.hybrid_search(
        collection_name=collection_name,
        reqs=requests,
        ranker=ranker,
        limit=limit,
        output_fields=output_fields,
    )


def build_book_filter_expr(book_names: List[str]) -> Optional[str]:
    """书名过滤表达式。"""
    if not book_names:
        return None
    names_literal = ", ".join(f'"{n}"' for n in book_names)
    return f"book_name in [{names_literal}]"


def build_content_type_filter_expr(
    book_names: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
) -> Optional[str]:
    """组合书名 + 内容类型的过滤表达式。"""
    clauses = []
    if book_names:
        names_literal = ", ".join(f'"{n}"' for n in book_names)
        clauses.append(f"book_name in [{names_literal}]")
    if content_types:
        types_literal = ", ".join(f'"{t}"' for t in content_types)
        clauses.append(f"content_type in [{types_literal}]")
    return " and ".join(clauses) if clauses else None
