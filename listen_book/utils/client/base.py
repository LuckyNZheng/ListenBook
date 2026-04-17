import logging
import os
import threading
from typing import TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class BaseClientManager:
    """双重检查锁单例基类。子类声明 _xxx_client + _xxx_lock，实现 _create_xxx。"""

    @staticmethod
    def _require_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"缺少必需的环境变量: {key}")
        return value

    @classmethod
    def _get_or_create(cls, attr_name: str, lock: threading.Lock, factory):
        instance = getattr(cls, attr_name, None)
        if instance is not None:
            return instance
        with lock:
            instance = getattr(cls, attr_name, None)
            if instance is not None:
                return instance
            instance = factory()
            setattr(cls, attr_name, instance)
            return instance
