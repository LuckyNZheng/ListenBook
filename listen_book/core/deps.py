"""依赖注入模块"""

from functools import cache

from listen_book.services.query_service import QueryService
from listen_book.services.import_service import ImportService


@cache
def get_import_service() -> ImportService:
    """获取导入服务单例"""
    return ImportService()


@cache
def get_query_service() -> QueryService:
    """获取查询服务单例"""
    return QueryService()
