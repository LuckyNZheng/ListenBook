"""存储客户端：Milvus / MongoDB。"""
import logging
import threading
from typing import Optional

from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymongo import MongoClient
from pymongo.database import Database

from listen_book.utils.client.base import BaseClientManager

load_dotenv()
logger = logging.getLogger(__name__)


class StorageClients(BaseClientManager):
    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    _mongo_db: Optional[Database] = None
    _mongo_lock = threading.Lock()

    # ---------- Milvus ----------
    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus)

    @classmethod
    def _create_milvus(cls) -> MilvusClient:
        uri = cls._require_env("MILVUS_URL")
        client = MilvusClient(uri=uri)
        logger.info(f"Milvus 初始化成功 (uri={uri})")
        return client

    # ---------- MongoDB ----------
    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create("_mongo_db", cls._mongo_lock, cls._create_mongo)

    @classmethod
    def _create_mongo(cls) -> Database:
        url = cls._require_env("MONGO_URL")
        db_name = cls._require_env("MONGO_DB_NAME")
        client = MongoClient(url)
        db = client[db_name]
        logger.info(f"MongoDB 初始化成功 (db={db_name})")
        return db
