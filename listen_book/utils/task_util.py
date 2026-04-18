"""任务状态追踪（进程内存储，用于单机教学场景）。"""
from collections import defaultdict
from typing import Any, Dict, List

TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_tasks_running: Dict[str, List[str]] = defaultdict(list)
_tasks_done: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)
_tasks_result: Dict[str, Dict[str, Any]] = defaultdict(dict)
_tasks_status: Dict[str, str] = {}

_NODE_CN = {
    # 导入
    "upload_file": "上传文件",
    "entry_node": "文件检查",
    "document_split_node": "文档切分",
    "book_recognition_node": "书名识别",
    "book_info_enrich_node": "信息补充",
    "embedding_chunks_node": "向量生成",
    "import_milvus_node": "写入向量库",
    # 查询
    "book_confirmed_node": "书名确认",
    "query_rewrite_node": "查询重写",
    "book_validation_node": "书名验证",
    "hybrid_vector_search_node": "向量检索",
    "hyde_vector_search_node": "HyDE检索",
    "rrf_merge_node": "结果融合",
    "reranker_node": "重排序",
    "answer_output_node": "答案生成",
    "__end__": "完成",
}


def _cn(name: str) -> str:
    return _NODE_CN.get(name, name)


def add_running_task(task_id: str, node_name: str) -> None:
    lst = _tasks_running[task_id]
    if node_name not in lst:
        lst.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    if node_name in _tasks_running[task_id]:
        _tasks_running[task_id].remove(node_name)
    done = _tasks_done[task_id]
    if node_name not in done:
        done.append(node_name)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    _tasks_duration[task_id][_cn(node_name)] = round(duration, 2)


def update_task_status(task_id: str, status: str) -> None:
    _tasks_status[task_id] = status


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def set_task_result(task_id: str, key: str, value: Any) -> None:
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: Any = "") -> Any:
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_info(task_id: str) -> Dict[str, Any]:
    return {
        "status": _tasks_status.get(task_id, ""),
        "running_list": [_cn(n) for n in _tasks_running.get(task_id, [])],
        "done_list": [_cn(n) for n in _tasks_done.get(task_id, [])],
        "durations": dict(_tasks_duration.get(task_id, {})),
    }
