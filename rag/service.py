"""RAG 问答服务（对齐 Java 参考项目的 RagService + RagController）。

职责：
- 文档上传：提取文本 → 语义分块 → 嵌入 → 写入 Milvus；
- 问答：向量检索（可选重排 / BM25 混合）→ 拼装上下文 → 复用本项目聊天模型生成答案；
- 管理：向量库状态、文档列表、按文档删除、整体清空。

聊天模型直接复用 app.models.create_llm（RagSettings 继承自 app.config.Settings），
因此 RAG 侧无需再维护一套模型配置。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import create_llm
from rag.config import RagSettings
from rag.document import DocumentChunk, create_document_chunks, extract_text
from rag.embeddings import build_embeddings
from rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from rag.retrieval import RetrievalService, format_context
from rag.retry import execute_with_retry, get_friendly_error_message
from rag.vectorstore import MilvusVectorStore


@dataclass
class UploadResult:
    """文档上传结果。"""

    filename: str
    document_id: str
    chunks: int
    total_chunks: int


class RagService:
    """RAG 问答服务：整合文档管理、检索与生成。"""

    def __init__(
        self,
        settings: RagSettings,
        vector_store: MilvusVectorStore | None = None,
        retrieval: RetrievalService | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        """
        Args:
            settings: RAG 配置（同时用于创建聊天模型）。
            vector_store: 向量库实例；为空则按配置创建。
            retrieval: 检索服务实例；为空则按配置创建。
            llm: 聊天模型实例；为空则延迟到首次问答时创建。
        """
        self.settings = settings
        if vector_store is None:
            vector_store = MilvusVectorStore(settings, build_embeddings(settings))
        self.vector_store = vector_store
        self.retrieval = retrieval or RetrievalService(settings, vector_store)

        self._llm = llm
        # 最近一次问答命中的文档块（供 CLI 展示溯源信息）
        self.last_retrieved: list[DocumentChunk] = []

    @property
    def llm(self) -> BaseChatModel:
        """聊天模型（延迟创建，避免仅做文档管理时也初始化模型）。"""
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    # ------------------------------------------------------------------ #
    # 文档上传
    # ------------------------------------------------------------------ #
    def upload(self, path: str | Path) -> UploadResult:
        """上传单个文档：提取 → 分块 → 嵌入 → 入库。

        Args:
            path: 文档路径（支持 PDF / DOCX / TXT / Markdown）。

        Returns:
            上传结果（文档 ID、分块数量、库内总量）。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 格式不支持或未提取到文本。
            RuntimeError: 嵌入或写入向量库失败。
        """
        file_path = Path(path)
        # 1. 提取文本
        text = extract_text(file_path)
        # 2. 语义分块（同一文档共用一个 document_id，便于隔离与删除）
        chunks = create_document_chunks(text, file_path.name, self.settings)
        # 3. 嵌入 + 写入向量库
        self.vector_store.add_chunks(chunks)
        # 4. 文档变更后让 BM25 索引在下次检索时重建
        self.retrieval.invalidate_bm25()

        return UploadResult(
            filename=file_path.name,
            # 空分块属防御性保护：正常流程 extract_text 已保证非空，不会触发
            document_id=chunks[0].document_id if chunks else "",
            chunks=len(chunks),
            total_chunks=self.vector_store.size(),
        )

    # ------------------------------------------------------------------ #
    # 问答
    # ------------------------------------------------------------------ #
    def _build_messages(self, context: str, question: str) -> list:
        """构造系统提示 + 用户消息。"""
        return [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(
                content=RAG_USER_TEMPLATE.format(context=context, question=question)
            ),
        ]

    def _prepare(self, query: str, document_id: str | None) -> list:
        """检索并构造消息；同时记录本次命中的文档块。"""
        chunks = self.retrieval.retrieve(
            query, self.settings.retrieval_top_k, document_id
        )
        self.last_retrieved = chunks
        context = format_context(chunks)
        return self._build_messages(context, query)

    def chat(self, query: str, document_id: str | None = None) -> str:
        """基于检索增强生成回答（非流式）。

        Args:
            query: 用户问题。
            document_id: 限定检索的文档 ID；为空则全库检索。

        Returns:
            模型回答；失败时返回中文友好提示（与 Java 端行为一致）。
        """
        try:
            messages = self._prepare(query, document_id)
            response = execute_with_retry(
                lambda: self.llm.invoke(messages),
                "聊天API调用",
                self.settings,
            )
            return str(response.content or "")
        except Exception as exc:  # noqa: BLE001 —— 与 Java 一致：失败返回友好提示
            return "抱歉，" + get_friendly_error_message(exc)

    def stream_chat(
        self, query: str, document_id: str | None = None
    ) -> Iterator[str]:
        """基于检索增强生成回答（流式，逐段产出文本）。"""
        try:
            messages = self._prepare(query, document_id)
        except Exception as exc:  # noqa: BLE001 —— 检索失败时直接给出提示
            yield "抱歉，" + get_friendly_error_message(exc)
            return

        try:
            for piece in self.llm.stream(messages):
                content = piece.content
                if content:
                    yield str(content)
        except Exception as exc:  # noqa: BLE001 —— 流式中断时追加提示，避免静默失败
            yield "\n\n[错误] " + get_friendly_error_message(exc)

    # ------------------------------------------------------------------ #
    # 管理
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        """向量库状态：分块总数 + 文档 ID 列表。"""
        return {
            "total_chunks": self.vector_store.size(),
            "documents": self.vector_store.get_all_document_ids(),
        }

    def list_documents(self) -> list[tuple[str, str, int]]:
        """文档列表：(文档 ID, 文件名, 分块数)。"""
        mapping = self.vector_store.get_document_id_to_filename_map()
        return [
            (doc_id, mapping.get(doc_id, "(未知)"), self.vector_store.size_of_document(doc_id))
            for doc_id in self.vector_store.get_all_document_ids()
        ]

    def document_status(self, document_id: str) -> int:
        """指定文档的分块数量。"""
        return self.vector_store.size_of_document(document_id)

    def delete_document(self, document_id: str) -> None:
        """删除指定文档的全部分块。"""
        self.vector_store.delete_document(document_id)
        self.retrieval.invalidate_bm25()

    def clear(self) -> None:
        """清空向量库（删除整个集合）。"""
        self.vector_store.clear_all()
        self.retrieval.invalidate_bm25()
