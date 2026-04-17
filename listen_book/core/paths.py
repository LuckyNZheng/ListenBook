import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_BASE_DIR = os.path.join(PROJECT_ROOT, "temp_data")
FRONT_PAGE_DIR = os.path.join(PROJECT_ROOT, "front_page")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def get_local_base_dir() -> str:
    os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
    return LOCAL_BASE_DIR


def get_temp_data_dir() -> str:
    """获取临时数据目录"""
    os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    return FRONT_PAGE_DIR


def get_data_dir() -> str:
    return DATA_DIR
