"""RAG 配置模块：在既有聊天配置之上，追加 RAG 相关配置项。

设计说明：
- 继承 app.config.Settings，因此 RAG 侧可直接复用聊天模型配置
  （即 create_llm(settings) 可以接收本模块的 RagSettings 实例）。
- 所有字段均可通过 .env / 环境变量覆盖，默认值对齐 Java 参考项目
  的 application.yml，便于与 Java 端共用同一套 Milvus / Redis 资源。
"""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings

# 检索重排方式：
#   keyword  —— 关键词匹配（出现次数 + 位置权重）
#   semantic —— 语义相似度（向量分数与关键词分加权）
RerankMethod = str


class RagSettings(Settings):
    """RAG 功能配置（字段名与 .env 中的变量名一一对应）。"""

    # ---------------- 嵌入模型（Embedding） ----------------
    # 留空 embedding_api_key / embedding_base_url 时会自动回退到 OPENAI_* 配置
    embedding_api_key: str = ""
    embedding_base_url: str | None = None
    embedding_model: str = "ep-20260420014217-l6bqr"
    embedding_dimension: int = 2048          # 嵌入向量维度，须与 Milvus 集合一致
    embedding_cache_enabled: bool = True     # 是否启用 Redis 嵌入缓存
    embedding_cache_ttl: int = 86400         # 缓存过期时间（秒），默认 24 小时

    # ---------------- Redis（嵌入向量缓存） ----------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # ---------------- Milvus 向量数据库 ----------------
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_token: str = ""                   # 未开启鉴权时留空
    milvus_collection_name: str = "rag_documents"
    milvus_id_field: str = "id"              # 主键字段名
    milvus_document_id_field: str = "document_id"  # 文档隔离字段名
    milvus_vector_field: str = "vector"      # 向量字段名
    milvus_content_field: str = "content"    # 正文字段名
    milvus_source_field: str = "source"      # 来源（文件名）字段名
    milvus_chunk_index_field: str = "chunk_index"  # 块序号字段名
    milvus_index_nlist: int = 128            # IVF_FLAT 索引 nlist 参数

    # ---------------- 文档分块 ----------------
    chunk_min_size: int = 200                # 最小块大小（字符）
    chunk_max_size: int = 800                # 最大块大小（字符）
    chunk_target_size: int = 500             # 目标块大小（字符）
    chunk_overlap: int = 100                 # 相邻块重叠大小（字符）

    # ---------------- 检索 ----------------
    retrieval_top_k: int = 3                 # 最终返回的文档块数量
    rerank_enabled: bool = True              # 是否启用重排序
    rerank_top_k: int = 5                    # 重排前召回的候选块数量
    rerank_final_top_k: int = 3              # 重排后返回的块数量
    rerank_method: RerankMethod = "keyword"  # keyword / semantic
    bm25_enabled: bool = False               # 是否启用 BM25 混合检索
    bm25_weight: float = 0.3                 # BM25 权重（向量权重为 1 - weight）
    bm25_top_k: int = 5                      # BM25 召回候选块数量

    # ---------------- 失败重试（指数退避） ----------------
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_initial_delay: int = 1000          # 初始延迟（毫秒）
    retry_max_delay: int = 10000             # 最大延迟（毫秒）
    retry_multiplier: float = 2.0            # 延迟倍数

    # ---------------- 派生属性 ----------------
    @property
    def milvus_uri(self) -> str:
        """Milvus 连接地址（与 Java 端一致：http://host:port）。"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def redis_url(self) -> str:
        """Redis 连接地址（redis://[:password@]host:port/db）。"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def effective_embedding_api_key(self) -> str:
        """嵌入 API Key：优先独立配置，否则复用聊天模型的 Key。"""
        return self.embedding_api_key or self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> str | None:
        """嵌入 Base URL：优先独立配置，否则复用聊天模型的地址。"""
        return self.embedding_base_url or self.openai_base_url


@lru_cache
def get_rag_settings() -> RagSettings:
    """返回全局唯一的 RagSettings 实例（带缓存，避免重复解析 .env）。"""
    return RagSettings()
