"""全局配置。用 dataclass + 环境变量，懒加载单例。"""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    llm_default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))
    book_recognition_model: str = field(
        default_factory=lambda: os.getenv("BOOK_RECOGNITION_MODEL", "")
    )
    llm_default_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))
    )

    # Embedding
    bge_m3_path: str = field(default_factory=lambda: os.getenv("BGE_M3_PATH", ""))
    bge_device: str = field(default_factory=lambda: os.getenv("BGE_DEVICE", "cpu"))
    bge_fp16: bool = field(
        default_factory=lambda: os.getenv("BGE_FP16", "False").lower() in ("true", "1")
    )

    # Reranker
    bge_reranker_path: str = field(default_factory=lambda: os.getenv("BGE_RERANKER_PATH", ""))
    bge_reranker_device: str = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_DEVICE", "cpu")
    )
    bge_reranker_fp16: bool = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_FP16", "False").lower() in ("true", "1")
    )

    # Milvus
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "lb_chunks"))
    book_name_collection: str = field(
        default_factory=lambda: os.getenv("BOOK_NAME_COLLECTION", "lb_book_names")
    )

    # MongoDB
    mongo_url: str = field(default_factory=lambda: os.getenv("MONGO_URL", ""))
    mongo_db_name: str = field(default_factory=lambda: os.getenv("MONGO_DB_NAME", "listten_book"))

    # 向量 & 切分
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    )
    max_chunk_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_CHUNK_LENGTH", "1500"))
    )
    min_chunk_length: int = field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_LENGTH", "300"))
    )

    # 检索
    hybrid_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_SEARCH_LIMIT", "8"))
    )
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "6"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
    )

    # 书名识别阈值
    book_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("BOOK_NAME_HIGH_CONFIDENCE", "0.7"))
    )
    book_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("BOOK_NAME_MID_CONFIDENCE", "0.45"))
    )
    book_name_score_gap: float = field(
        default_factory=lambda: float(os.getenv("BOOK_NAME_SCORE_GAP", "0.15"))
    )
    book_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("BOOK_NAME_MAX_OPTIONS", "3"))
    )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
