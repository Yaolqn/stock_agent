"""Milvus 向量存储封装（对齐 Java 参考项目 VectorStoreService）。

单集合架构：所有文档共用一个集合，通过 document_id 字段做逻辑隔离。
字段与 Java 端保持一致，因此 Python 与 Java 可以共用同一个 Milvus 集合：
    id          VarChar         主键（文档块 ID）
    document_id VarChar(128)    文档隔离字段
    vector      FloatVector(dim) 嵌入向量
    content     VarChar(65535)  正文
    source      VarChar(512)    来源文件名
    chunk_index Int64           块序号
索引：IVF_FLAT + COSINE，nlist 可配置。

实现说明（LangChain 1.x / pymilvus 3.x 兼容）：
langchain-milvus 0.4.x 底层已切换到 pymilvus 3.x 的 MilvusClient API，
旧版 `Collection.query()/delete()/flush()/drop()` 的 ORM 集合对象已移除，
store.col 仅保留"集合是否存在"与 schema/index 视图。因此本模块的查询、
删除、清空、刷盘一律通过 `store.client`（MilvusClient）完成。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus import DataType

from rag.config import RagSettings
from rag.document import DocumentChunk
from rag.retry import execute_with_retry

# 查询时单次拉取的最大行数（与 Java 端一致）
_QUERY_LIMIT = 10000


class MilvusVectorStore:
    """Milvus 向量库封装：增删查清 + 文档级隔离。"""

    def __init__(self, settings: RagSettings, embeddings: Embeddings) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self._store = self._build_store()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _build_store(self) -> Milvus:
        """构造 langchain-milvus 向量库实例（集合不存在时会在首次写入时创建）。"""
        s = self.settings

        connection_args: dict = {"uri": s.milvus_uri}
        if s.milvus_token:
            connection_args["token"] = s.milvus_token

        # 元数据字段显式声明类型，避免 langchain 按首条数据推断出过大的字段
        metadata_schema = {
            s.milvus_document_id_field: {
                "dtype": DataType.VARCHAR,
                "kwargs": {"max_length": 128},
            },
            s.milvus_source_field: {
                "dtype": DataType.VARCHAR,
                "kwargs": {"max_length": 512},
            },
            s.milvus_chunk_index_field: {"dtype": DataType.INT64},
        }

        return Milvus(
            embedding_function=self.embeddings,
            collection_name=s.milvus_collection_name,
            connection_args=connection_args,
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": s.milvus_index_nlist},
            },
            auto_id=False,  # 主键由本端生成（与 Java 一致）
            primary_field=s.milvus_id_field,
            text_field=s.milvus_content_field,
            vector_field=s.milvus_vector_field,
            metadata_schema=metadata_schema,
        )

    @property
    def _collection(self):
        """底层集合视图；集合尚未创建时为 None（用于"是否存在"判定）。

        注意：langchain-milvus 0.4.x 中该对象仅提供 schema/index 元数据，
        数据操作（query/delete/flush/drop）必须走 `self._client`。
        """
        return self._store.col

    @property
    def _client(self):
        """底层 pymilvus 3.x MilvusClient（数据操作统一入口）。"""
        return self._store.client

    def _filter_expr(self, document_id: str) -> str:
        """构造文档隔离过滤表达式。"""
        return f"{self.settings.milvus_document_id_field} == '{document_id}'"

    def _query(self, expr: str, output_fields: list[str]) -> list[dict]:
        """按表达式查询（集合不存在时返回空列表）。

        pymilvus 3.x：查询走 MilvusClient.query()，返回 list[dict]，
        每个 dict 的键为 schema 中的字段名。
        """
        if self._collection is None:
            return []
        try:
            return self._client.query(
                self.settings.milvus_collection_name,
                filter=expr,
                output_fields=output_fields,
                limit=_QUERY_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001 —— 查询失败按空结果处理
            print(f"  [向量库] 查询失败：{exc}")
            return []

    def _flush(self) -> None:
        """刷盘，保证 /status 统计与检索立即可见。"""
        if self._collection is not None:
            self._client.flush(self.settings.milvus_collection_name)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """把文档块写入向量库（含嵌入生成，带重试与缓存）。

        Args:
            chunks: 待写入的文档块列表。

        Raises:
            RuntimeError: 嵌入生成或写入失败。
        """
        if not chunks:
            return

        s = self.settings
        # 提取文本内容、ID、元数据
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        metadatas = [
            {
                s.milvus_document_id_field: chunk.document_id or "default",
                s.milvus_source_field: chunk.source,
                s.milvus_chunk_index_field: chunk.chunk_index,
            }
            for chunk in chunks
        ]

        # 先生成向量（带指数退避重试 + Redis 缓存），再写入，便于精确控制重试
        try:
            vectors = execute_with_retry(
                lambda: self.embeddings.embed_documents(texts),
                "嵌入向量生成",
                s,
            )
        except Exception as exc:  # noqa: BLE001 —— 统一转成中文友好提示
            raise RuntimeError(f"嵌入向量生成失败：{exc}") from exc

        try:
            self._store.add_embeddings(
                texts=texts,
                embeddings=vectors,
                metadatas=metadatas,
                ids=ids,
            )
            # 刷盘，保证后续 /status 统计与检索立即可见
            self._flush()
        except Exception as exc:  # noqa: BLE001 —— 写入失败向上抛出，避免假成功
            raise RuntimeError(f"写入向量库失败：{exc}") from exc

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[DocumentChunk]:
        """向量相似度检索（可选文档级隔离）。

        Args:
            query: 查询文本。
            top_k: 召回数量。
            document_id: 传入则只在该文档内检索，为空则全局检索。

        Returns:
            按相似度降序排列的文档块列表。
        """
        if self._collection is None:
            return []

        expr = self._filter_expr(document_id) if document_id else None
        try:
            pairs = self._store.similarity_search_with_score(query, k=top_k, expr=expr)
        except Exception as exc:  # noqa: BLE001 —— 检索失败按无结果处理
            print(f"  [向量库] 检索失败：{exc}")
            return []

        s = self.settings
        results: list[DocumentChunk] = []
        for document, score in pairs:
            meta = document.metadata or {}
            results.append(
                DocumentChunk(
                    id=str(meta.get(s.milvus_id_field, "")),
                    document_id=str(meta.get(s.milvus_document_id_field, "")),
                    content=document.page_content,
                    source=str(meta.get(s.milvus_source_field, "")),
                    chunk_index=int(meta.get(s.milvus_chunk_index_field, 0) or 0),
                    similarity=float(score),
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # 删除 / 统计
    # ------------------------------------------------------------------ #
    def delete_document(self, document_id: str) -> None:
        """删除某个文档的全部分块（保留集合）。

        pymilvus 3.x：删除走 MilvusClient.delete()，filter 传过滤表达式。
        """
        if not document_id or self._collection is None:
            return
        self._client.delete(
            self.settings.milvus_collection_name, filter=self._filter_expr(document_id)
        )
        self._flush()

    def clear_all(self) -> None:
        """危险操作：删除整个集合，并重建向量库句柄以便下次写入时重新建表。

        用 store.drop()（而非直接 drop_collection）可一并清理 langchain-milvus
        内部与 pymilvus 的 schema 缓存，避免重新建集合时 schema 冲突。
        """
        if self._collection is None:
            return
        self._store.drop()
        self._store = self._build_store()

    def size(self) -> int:
        """集合中的文档块总数。"""
        if self._collection is None:
            return 0
        try:
            stats = self._store.client.get_collection_stats(
                self.settings.milvus_collection_name
            )
            return int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001 —— 统计失败按 0 处理
            return 0

    def size_of_document(self, document_id: str) -> int:
        """某个文档的分块数量。"""
        if not document_id:
            return 0
        rows = self._query(
            self._filter_expr(document_id), [self.settings.milvus_id_field]
        )
        return len(rows)

    def get_all_document_ids(self) -> list[str]:
        """库中所有文档 ID（去重，保持首次出现顺序）。"""
        s = self.settings
        rows = self._query(f"{s.milvus_id_field} != ''", [s.milvus_document_id_field])
        ids = [row.get(s.milvus_document_id_field) for row in rows]
        return list(dict.fromkeys(i for i in ids if i))

    def get_document_id_to_filename_map(self) -> dict[str, str]:
        """文档 ID → 文件名 的映射。"""
        s = self.settings
        rows = self._query(
            f"{s.milvus_id_field} != ''",
            [s.milvus_document_id_field, s.milvus_source_field],
        )
        mapping: dict[str, str] = {}
        for row in rows:
            doc_id = row.get(s.milvus_document_id_field)
            source = row.get(s.milvus_source_field)
            if doc_id and source and doc_id not in mapping:
                mapping[doc_id] = source
        return mapping

    def get_all_chunks(self) -> list[DocumentChunk]:
        """拉取所有文档块（供 BM25 构建索引，数据量大时慎用）。"""
        s = self.settings
        rows = self._query(
            f"{s.milvus_id_field} != ''",
            [
                s.milvus_id_field,
                s.milvus_document_id_field,
                s.milvus_content_field,
                s.milvus_source_field,
                s.milvus_chunk_index_field,
            ],
        )
        return [
            DocumentChunk(
                id=str(row.get(s.milvus_id_field, "")),
                document_id=str(row.get(s.milvus_document_id_field, "")),
                content=str(row.get(s.milvus_content_field, "")),
                source=str(row.get(s.milvus_source_field, "")),
                chunk_index=int(row.get(s.milvus_chunk_index_field, 0) or 0),
            )
            for row in rows
        ]


def describe_vector_store(settings: RagSettings) -> str:
    """生成向量库配置的文字描述（供 CLI 展示）。"""
    return (
        f"milvus={settings.milvus_uri} | collection={settings.milvus_collection_name}"
        f" | dimension={settings.embedding_dimension}"
        f" | index=IVF_FLAT/COSINE(nlist={settings.milvus_index_nlist})"
    )
