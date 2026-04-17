"""Redis 缓存工具"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any, List

import redis

from listen_book.core.config import get_settings

logger = logging.getLogger(__name__)

# Redis 客户端单例
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端单例"""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        try:
            _redis_client.ping()
            logger.info("Redis 连接成功")
        except redis.ConnectionError as e:
            logger.warning(f"Redis 连接失败: {e}")
            _redis_client = None
            raise
    return _redis_client


def close_redis_client() -> None:
    """关闭 Redis 客户端"""
    global _redis_client
    if _redis_client is not None:
        _redis_client.close()
        _redis_client = None


def build_cache_key(query: str, book_names: List[str] = None) -> str:
    """构建缓存键

    使用原问题 + 识别书名作为缓存键

    Args:
        query: 用户原始问题
        book_names: 识别到的书名列表

    Returns:
        缓存键字符串
    """
    # 构建键的原始数据
    key_data = {
        "query": query.strip().lower(),  # 统一小写，忽略大小写差异
        "book_names": sorted(book_names) if book_names else [],  # 排序保证一致性
    }

    # 生成 MD5 哈希作为键
    key_str = json.dumps(key_data, ensure_ascii=False, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()

    return f"lb:query:{key_hash}"


def get_cached_answer(query: str, book_names: List[str] = None) -> Optional[Dict[str, Any]]:
    """从缓存获取答案

    Args:
        query: 用户原始问题
        book_names: 识别到的书名列表

    Returns:
        缓存的答案数据，未命中返回 None
    """
    settings = get_settings()

    if not settings.cache_enabled:
        return None

    try:
        client = get_redis_client()
        cache_key = build_cache_key(query, book_names)

        cached_data = client.get(cache_key)
        if cached_data:
            logger.info(f"缓存命中: {cache_key}")
            return json.loads(cached_data)

        logger.info(f"缓存未命中: {cache_key}")
        return None

    except Exception as e:
        logger.warning(f"读取缓存失败: {e}")
        return None


def set_cached_answer(
    query: str,
    book_names: List[str] = None,
    answer: str = "",
    sources: List[Dict] = None,
    rewritten_query: str = "",
) -> bool:
    """写入缓存

    Args:
        query: 用户原始问题
        book_names: 识别到的书名列表
        answer: 答案内容
        sources: 来源信息
        rewritten_query: 重写后的查询

    Returns:
        是否写入成功
    """
    settings = get_settings()

    if not settings.cache_enabled:
        return False

    try:
        client = get_redis_client()
        cache_key = build_cache_key(query, book_names)

        cache_data = {
            "answer": answer,
            "sources": sources or [],
            "rewritten_query": rewritten_query,
            "book_names": book_names or [],
        }

        client.setex(
            cache_key,
            settings.cache_expire_seconds,
            json.dumps(cache_data, ensure_ascii=False),
        )

        logger.info(f"缓存写入: {cache_key}")
        return True

    except Exception as e:
        logger.warning(f"写入缓存失败: {e}")
        return False


def clear_query_cache() -> int:
    """清除所有查询缓存"""
    try:
        client = get_redis_client()
        keys = client.keys("lb:query:*")
        if keys:
            deleted = client.delete(*keys)
            logger.info(f"清除缓存: {deleted} 条")
            return deleted
        return 0
    except Exception as e:
        logger.warning(f"清除缓存失败: {e}")
        return 0