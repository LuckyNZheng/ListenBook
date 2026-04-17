"""导入相关 Schema 定义"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ContentTypeEnum(str, Enum):
    """内容类型枚举"""
    AUDIO_BOOK_INFO = "有声书信息"
    BOOK_INTRO = "书籍简介"
    AUTHOR_INTRO = "作者介绍"
    LISTENING_NOTE = "听书笔记"
    RECOMMEND_MATERIAL = "推荐运营资料"
    USER_COMMENT = "用户评论摘要"
    FAQ = "常见问答"


class UploadRequest(BaseModel):
    """上传请求"""
    file_path: str = Field(..., description="文件路径")
    file_type: str = Field("md", description="文件类型：pdf 或 md")
    book_name: Optional[str] = Field(None, description="书名（可选，不传则自动识别）")
    author: Optional[str] = Field(None, description="作者名（可选）")
    content_type: ContentTypeEnum = Field(
        ContentTypeEnum.BOOK_INTRO, description="内容类型"
    )
    category: Optional[str] = Field(None, description="类别/标签")
    audio_duration: Optional[str] = Field(None, description="有声书时长")
    source_url: Optional[str] = Field(None, description="来源URL")


class UploadResponse(BaseModel):
    """上传响应"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")
    file_name: str = Field("", description="文件名")


class UploadStatusResponse(BaseModel):
    """上传状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="状态：pending/processing/completed/failed")
    running_list: List[str] = Field(default_factory=list, description="正在执行的节点")
    done_list: List[str] = Field(default_factory=list, description="已完成的节点")
    error: str = Field("", description="错误信息")
    chunk_count: int = Field(0, description="切片数量")


class ChunkInfo(BaseModel):
    """切片信息"""
    chunk_id: int = Field(..., description="切片ID")
    content: str = Field(..., description="切片内容")
    book_name: str = Field("", description="书名")
    author: str = Field("", description="作者")
    content_type: str = Field("", description="内容类型")
    entry_name: str = Field("", description="条目名称")
    category: str = Field("", description="类别")
    source_file: str = Field("", description="来源文件名")