"""MongoDB 对话历史工具。"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import DESCENDING
from pymongo.collection import Collection

from listen_book.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _collection() -> Collection:
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    book_names: Optional[List[str]] = None,
    message_id: Optional[str] = None,
) -> str:
    doc = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "book_names": book_names or [],
        "ts": datetime.now().timestamp(),
    }
    coll = _collection()
    if message_id:
        coll.update_one({"_id": ObjectId(message_id)}, {"$set": doc})
        return message_id
    return str(coll.insert_one(doc).inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        cursor = (
            _collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"读取历史失败: {e}")
        return []


def clear_history(session_id: str) -> int:
    try:
        return _collection().delete_many({"session_id": session_id}).deleted_count
    except Exception as e:
        logger.error(f"清空历史失败: {e}")
        return 0
