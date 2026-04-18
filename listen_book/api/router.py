"""API 路由"""

import os
import asyncio
import uuid
import logging

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from listen_book.core.paths import get_front_page_dir, get_temp_data_dir
from listen_book.schemas.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from listen_book.schemas.upload_schema import UploadResponse, UploadStatusResponse
from listen_book.services.query_service import QueryService
from listen_book.services.import_service import ImportService
from listen_book.utils.sse_util import create_sse_queue, sse_generator
from listen_book.utils.task_util import get_task_info, get_task_result, TASK_STATUS_FAILED
from listen_book.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def get_query_service() -> QueryService:
    """获取查询服务"""
    return QueryService()


def get_import_service() -> ImportService:
    """获取导入服务"""
    return ImportService()


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(title="听书知识库", version="1.0")

    # 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 静态文件
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front_page", StaticFiles(directory=front_page_dir), name="front_page")

    register_router(app)
    return app


def register_router(app: FastAPI) -> None:
    """注册路由"""

    @app.get("/")
    def index():
        return {"message": "听书知识库 API", "version": "1.0"}

    # ---------- 查询接口 ----------
    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ):
        """处理查询请求"""
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        if request.is_stream:
            create_sse_queue(task_id)
            background_tasks.add_task(
                service.run_query_graph,
                session_id,
                task_id,
                request.query,
                request.is_stream,
            )
            return StreamSubmitResponse(
                message="查询请求已提交",
                session_id=session_id,
                task_id=task_id,
            )
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                service.run_query_graph,
                session_id,
                task_id,
                request.query,
                request.is_stream,
            )
            answer = service.get_task_result(task_id)
            return QueryResponse(
                message="查询完成",
                session_id=session_id,
                answer=answer,
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """流式输出"""
        return StreamingResponse(
            content=sse_generator(task_id, request),
            media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str,
        limit: int = 50,
        service: QueryService = Depends(get_query_service),
    ):
        """获取历史对话"""
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取历史失败: {e}")

    @app.delete("/history/{session_id}")
    async def clear_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
    ):
        """清空历史"""
        count = service.clear_history(session_id)
        return {"message": "历史已清空", "deleted_count": count}

    @app.get("/status/{task_id}")
    async def get_task_status_api(task_id: str):
        """获取任务状态"""
        info = get_task_info(task_id)
        info["answer"] = get_task_result(task_id, "answer")
        info["error"] = get_task_result(task_id, "error")
        if info.get("status") == TASK_STATUS_FAILED and not info.get("error"):
            info["error"] = "任务执行失败"
        return info

    # ---------- 导入接口 ----------
    @app.post("/upload/file", response_model=UploadResponse)
    async def upload_file(
        background_tasks: BackgroundTasks,
        service: ImportService = Depends(get_import_service),
        file: UploadFile = File(...),
        content_type: str = Form("书籍简介"),
        book_name: str = Form(None),
        author: str = Form(None),
        category: str = Form(None),
        audio_duration: str = Form(None),
        source_url: str = Form(None),
    ):
        """处理文件上传请求"""
        # 检查文件类型
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.md', '.txt', '.markdown']:
            raise HTTPException(status_code=400, detail="只支持 Markdown 文件 (.md, .txt, .markdown)")

        # 保存到临时目录
        task_id = service.generate_task_id()
        temp_dir = os.path.join(get_temp_data_dir(), task_id)
        os.makedirs(temp_dir, exist_ok=True)

        # 保存文件名（保留原始名称）
        safe_name = os.path.basename(file.filename)
        temp_path = os.path.join(temp_dir, safe_name)

        # 写入文件
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        logger.info(f"文件已保存: {temp_path}, 大小: {len(content)} bytes")

        # 启动导入任务
        background_tasks.add_task(
            service.run_import_graph,
            task_id,
            temp_path,
            book_name or "",
            author or "",
            content_type,
            category or "",
            audio_duration or "",
            source_url or "",
        )

        return UploadResponse(
            message="文件上传成功，正在处理",
            task_id=task_id,
            file_name=file.filename,
        )

    @app.get("/upload/status/{task_id}", response_model=UploadStatusResponse)
    async def upload_status(
        task_id: str,
        service: ImportService = Depends(get_import_service),
    ):
        """获取上传状态"""
        info = service.get_task_status(task_id)
        return UploadStatusResponse(
            task_id=task_id,
            status=info.get("status", ""),
            running_list=info.get("running_list", []),
            done_list=info.get("done_list", []),
            error=info.get("error", ""),
            chunk_count=info.get("chunk_count", 0),
        )

    # ---------- 书籍管理接口 ----------
    @app.get("/books")
    async def get_books():
        """获取所有书籍列表"""
        try:
            from listen_book.core.config import get_settings
            settings = get_settings()
            client = StorageClients.get_milvus_client()
            collection_name = settings.chunks_collection

            # 查询所有唯一的书名
            results = client.query(
                collection_name=collection_name,
                filter="",  # 无过滤条件，获取所有数据
                output_fields=["book_name", "author", "category", "content_type", "audio_duration", "source_file"],
                limit=10000,  # 设置较大的限制
            )

            # 按书名去重并统计
            books = {}
            for item in results:
                book_name = item.get("book_name", "")
                if not book_name:
                    continue

                if book_name not in books:
                    books[book_name] = {
                        "book_name": book_name,
                        "author": item.get("author", ""),
                        "category": item.get("category", ""),
                        "audio_duration": item.get("audio_duration", ""),
                        "content_types": set(),
                        "chunk_count": 0,
                    }
                books[book_name]["content_types"].add(item.get("content_type", ""))
                books[book_name]["chunk_count"] += 1

            # 转换为列表
            book_list = []
            for book in books.values():
                book["content_types"] = list(book["content_types"])
                book_list.append(book)

            # 按书名排序
            book_list.sort(key=lambda x: x["book_name"])

            return {
                "total": len(book_list),
                "books": book_list
            }
        except Exception as e:
            logger.error(f"查询书籍列表失败: {e}")
            raise HTTPException(status_code=500, detail=f"查询书籍列表失败: {e}")