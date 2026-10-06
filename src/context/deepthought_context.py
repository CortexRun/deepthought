from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing import List
from enum import Enum
from langchain_core.language_models import BaseChatModel

from psycopg2.extensions import connection  # type: ignore
import psycopg2  # type: ignore

import logging
import httpx

from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlmApiModel(str, Enum):
    GRANITE_350 = "granite4:350m"
    QWEN3_1_7 = "qwen3:1.7b-q4_K_M"
    GRANITE4_3B = "granite4:3b"  # verified: runs on the Jetson Orin Nano 8 GB
    GRANITE3_2B = "granite3-dense:2b"
    LLAMA32_3B = "llama3.2:3b"
    PHI_MINI = "phi4-mini:latest"
    QWEN2_1_5 = "qwen2:1.5b"
    NOMI_EMBD = "nomic-embed-text:latest"
    GRANITE3_3_2B = "granite3.3:2b"  


class DeepthoughtExecContext:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
    ):
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=60.0)

    def create_llm(
        self,
        model: str = LlmApiModel.GRANITE4_3B,
        temperature: float = 0.2,
        num_predict: int = 2048,
        structured_output_schema=None,
    ) -> BaseChatModel:
        """Creates a langchain LLM with the given parameters

        args:
            model
        """
        llm = ChatOllama(
            base_url=self.base_url,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
        )

        if structured_output_schema is not None:
            llm = llm.with_structured_output(structured_output_schema)  # type: ignore

        return llm

    def create_multiple_embeddings(self, text: List[str]) -> List[List[float]]:
        """Creates multiple embeddings for the provided text"""
        client = OllamaEmbeddings(
            base_url=self.base_url,
            model=LlmApiModel.NOMI_EMBD,
        )
        response = client.embed_documents(texts=text)
        return response

    def establish_db_connection(
        self, host: str, port: int, dbname: str, user: str, password: str
    ) -> connection:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        return conn
