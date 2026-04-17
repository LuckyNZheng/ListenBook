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
    schema.add_field("category", DataType.VARCHAR, max_length=200)
    schema.add_field("audio_duration", DataType.VARCHAR, max_length=50)
    schema.add_field("source_file", DataType.VARCHAR, max_length=500)    # 新增补充字段
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
    schema.add_field("category", DataType.VARCHAR, max_length=200)
    schema.add_field("target_audience", DataType.VARCHAR, max_length=200)
    schema.add_field("audio_duration", DataType.VARCHAR, max_length=200)

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


def build_combined_filter_expr(
    book_names: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
    intent: Optional[str] = None,
) -> Optional[str]:
    """构建组合过滤表达式，支持书名、类别、意图等多条件过滤。"""
    clauses = []

    # 书名过滤
    if book_names:
        names_literal = ", ".join(f'"{n}"' for n in book_names)
        clauses.append(f"book_name in [{names_literal}]")

    # 类别过滤
    if categories:
        categories_literal = ", ".join(f'"{c}"' for c in categories)
        clauses.append(f"category in [{categories_literal}]")

    # 内容类型过滤（直接指定）
    if content_types:
        types_literal = ", ".join(f'"{t}"' for t in content_types)
        clauses.append(f"content_type in [{types_literal}]")

    # 意图相关的内容类型偏好
    if intent == "recommend" and not content_types:
        # 推荐意图优先检索推荐相关内容
        clauses.append(f"content_type in [\"书籍简介\", \"推荐运营资料\", \"用户评论摘要\"]")
    elif intent == "detail" and not content_types:
        # 详情意图优先检索书籍详情相关内容
        clauses.append(f"content_type in [\"书籍简介\", \"作者介绍\", \"有声书信息\", \"常见问答\"]")

    return " and ".join(clauses) if clauses else None


def search_book_name_collection(
    client: MilvusClient,
    collection_name: str,
    book_name: str,
    bge_m3: Any,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """在书名集合中搜索，用于书名对齐验证。"""
    try:
        # 生成书名向量
        vectors = bge_m3.encode([book_name], return_dense=True, return_sparse=True)
        dense_vector = vectors["dense"][0]
        sparse_vector = vectors["sparse"][0]

        # 构建检索请求
        requests = build_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )

        # 输出字段
        output_fields = ["book_name", "author", "category"]

        # 执行检索
        results = execute_hybrid_search(
            client=client,
            collection_name=collection_name,
            requests=requests,
            output_fields=output_fields,
            limit=limit,
        )

        # 解析结果
        hits = []
        if results:
            for hit_list in results:
                for hit in hit_list:
                    entity = hit.get("entity", {})
                    hits.append({
                        "book_name": entity.get("book_name", ""),
                        "author": entity.get("author", ""),
                        "category": entity.get("category", ""),
                        "score": hit.get("distance", 0.0),
                    })

        return hits
    except Exception as e:
        logger.warning(f"书名检索失败: {e}")
        return []
