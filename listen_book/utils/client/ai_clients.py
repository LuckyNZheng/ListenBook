"""AI 模型客户端：LLM（OpenAI 兼容）/ BGE-M3 / Reranker。"""
import logging
import threading
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from listen_book.utils.client.base import BaseClientManager

load_dotenv()
logger = logging.getLogger(__name__)


class AIClients(BaseClientManager):
    _llm_text_client: Optional[ChatOpenAI] = None
    _llm_text_lock = threading.Lock()

    _llm_json_client: Optional[ChatOpenAI] = None
    _llm_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _reranker_client = None
    _reranker_lock = threading.Lock()

    # ---------- LLM ----------
    @classmethod
    def get_llm_client(cls, json_mode: bool = False) -> ChatOpenAI:
        if json_mode:
            return cls._get_or_create(
                "_llm_json_client", cls._llm_json_lock, lambda: cls._create_llm(True)
            )
        return cls._get_or_create(
            "_llm_text_client", cls._llm_text_lock, lambda: cls._create_llm(False)
        )

    @classmethod
    def _create_llm(cls, json_mode: bool) -> ChatOpenAI:
        api_key = cls._require_env("OPENAI_API_KEY")
        base_url = cls._require_env("OPENAI_API_BASE")
        model_name = cls._require_env("LLM_DEFAULT_MODEL")

        model_kwargs = {}
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}

        client = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=SecretStr(api_key),
            temperature=0,
            model_kwargs=model_kwargs,
        )
        logger.info(f"LLM 客户端初始化成功 (model={model_name}, json={json_mode})")
        return client

    # ---------- BGE-M3 Embedding ----------
    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3)

    @classmethod
    def _create_bge_m3(cls) -> BGEM3EmbeddingFunction:
        model_path = cls._require_env("BGE_M3_PATH")
        import os

        device = os.getenv("BGE_DEVICE", "cpu")
        fp16 = os.getenv("BGE_FP16", "False").lower() in ("true", "1")
        client = BGEM3EmbeddingFunction(
            model_name=model_path, device=device, use_fp16=fp16, return_sparse=True
        )
        logger.info(f"BGE-M3 初始化成功 (device={device})")
        return client

    # ---------- Reranker ----------
    @classmethod
    def get_reranker_client(cls):
        return cls._get_or_create("_reranker_client", cls._reranker_lock, cls._create_reranker)

    @classmethod
    def _create_reranker(cls):
        from FlagEmbedding import FlagReranker
        import os

        model_path = cls._require_env("BGE_RERANKER_PATH")
        device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
        fp16 = os.getenv("BGE_RERANKER_FP16", "False").lower() in ("true", "1")
        client = FlagReranker(model_name_or_path=model_path, device=device, use_fp16=fp16)
        logger.info(f"Reranker 初始化成功 (device={device})")
        return client


if __name__ == '__main__':
    llm = AIClients.get_llm_client(json_mode=True)
    res= llm.invoke(["你是什么模型,请回我：{nzmodel：你的模型名} "])
    print(res)
