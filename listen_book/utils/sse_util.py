"""SSE 事件推送（流式输出）。"""
import asyncio
import json
import logging
import queue
from typing import Any, AsyncGenerator, Dict

from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"


_streams: Dict[str, queue.Queue] = {}


def create_sse_queue(task_id: str) -> queue.Queue:
    q = queue.Queue()
    _streams[task_id] = q
    return q


def get_sse_queue(task_id: str):
    return _streams.get(task_id)


def remove_sse_queue(task_id: str) -> None:
    _streams.pop(task_id, None)


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]) -> None:
    q = _streams.get(task_id)
    if q:
        q.put({"event": event, "data": data})


def _pack(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator[str, None]:
    q = _streams.get(task_id)
    if q is None:
        return

    loop = asyncio.get_event_loop()
    try:
        while True:
            if await request.is_disconnected():
                return
            try:
                msg = await loop.run_in_executor(None, q.get, True, 1)
                yield _pack(msg.get("event"), msg.get("data"))
                if msg.get("event") == SSEEvent.FINAL:
                    return
            except queue.Empty:
                continue
    except (ConnectionResetError, BrokenPipeError):
        return
    except asyncio.CancelledError:
        raise
    finally:
        remove_sse_queue(task_id)
