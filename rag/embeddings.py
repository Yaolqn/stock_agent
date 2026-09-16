"""嵌入模型工厂：创建「带 Redis 缓存」的文本嵌入模型。

要点：
- 使用火山引擎 Ark 的「多模态嵌入」接口（与 Java 参考项目一致）：
      POST {base_url}/embeddings/multimodal
      body: {"model": "<推理端点ID>", "input": [{"type": "text", "text": "..."}]}
      resp: {"data": {"embedding": [...]}}
  该端点只支持多模态接口，不能用 OpenAI 兼容的 /embeddings 接口调用
  （否则会报 "the requested model does not support this api"）。
- 该接口「一次请求只返回一个向量」，因此批量嵌入按条循环（与 Java
  EmbeddingService.generateEmbeddings 行为一致）。
- 嵌入缓存复用 Java 参考项目的 Redis，缓存键与 Java 端保持一致：
      embedding:<md5(model + ":" + text)>
- Redis 不可用时自动降级为「不使用缓存」，不影响主流程。
"""

from __future__ import annotations

import hashlib

import requests
# LangChain 1.x：CacheBackedEmbeddings 已从 langchain 顶层迁出，改由
# langchain_classic 提供（旧模块在 1.x 中仅保留空壳，导入会失败）。
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings

from rag.config import RagSettings

# 单次嵌入请求的超时时间（秒）
_EMBED_TIMEOUT = 60.0


class ArkMultiModalEmbeddings(Embeddings):
    """火山引擎 Ark 多模态嵌入端点客户端。

    Ark 的 /embeddings/multimodal 一次请求只产生一个向量，
    所以文档批量嵌入需要逐条请求（与 Java 端实现保持一致）。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = _EMBED_TIMEOUT,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed(self, text: str) -> list[float]:
        """调用多模态嵌入接口，返回单个文本的向量。"""
        response = requests.post(
            f"{self.base_url}/embeddings/multimodal",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": [{"type": "text", "text": text}]},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"嵌入接口调用失败（HTTP {response.status_code}）：{response.text[:200]}"
            )

        data = response.json().get("data")
        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not embedding:
            raise RuntimeError("嵌入接口返回结果中缺少 data.embedding")
        return [float(value) for value in embedding]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成向量（逐条请求，顺序与入参一致）。"""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """生成单条文本的向量。"""
        return self._embed(text)


def _md5_key_encoder(model: str):
    """构造与 Java 端一致的缓存键编码器：embedding:<md5(model:text)>。

    Args:
        model: 嵌入模型名（参与缓存键计算，不同模型缓存互不串扰）。

    Returns:
        编码函数（接收文本，返回形如 embedding:<md5> 的缓存键字符串）。
    """

    def _encode(text: str) -> str:
        """把一段文本映射为 Redis 缓存键。"""
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
        return f"embedding:{digest}"

    return _encode


def _build_base_embeddings(settings: RagSettings) -> Embeddings:
    """创建底层嵌入模型（Ark 多模态嵌入端点）。"""
    return ArkMultiModalEmbeddings(
        model=settings.embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url or "",
    )


def _try_build_redis_store(settings: RagSettings):
    """尝试创建 Redis 缓存后端；不可用时返回 None（降级为无缓存）。"""
    if not settings.embedding_cache_enabled:
        return None
    try:
        import redis
        from langchain_community.storage import RedisStore

        # 主动探测一次连通性，避免运行期才暴露 Redis 故障
        redis.Redis.from_url(settings.redis_url).ping()
        return RedisStore(redis_url=settings.redis_url, ttl=settings.embedding_cache_ttl)
    except Exception as exc:  # noqa: BLE001 —— 缓存属可选能力，失败即降级
        print(f"  [缓存] Redis 不可用，嵌入缓存已禁用（{exc}）")
        return None

# Embeddings 是文本嵌入模型接口，定义了嵌入方法 embed_documents 和 embed_query
# 这里返回的是 CacheBackedEmbeddings，它实现了嵌入模型接口，同时添加了缓存功能
def build_embeddings(settings: RagSettings) -> Embeddings:
    """创建嵌入模型（带 Redis 缓存，缓存不可用时自动降级）。"""
    base = _build_base_embeddings(settings)
    store = _try_build_redis_store(settings)
    if store is None:
        return base

    print(f"  [缓存] 已启用 Redis 嵌入缓存，TTL={settings.embedding_cache_ttl}s")
    # 自定义 key_encoder 时必须传 namespace=""（前缀已在编码器内部处理）
    # from_bytes_store 会自动将嵌入结果转换为 bytes 类型，再写入 Redis，避免直接写入 str
    return CacheBackedEmbeddings.from_bytes_store(
        base,
        store,
        namespace="",
        key_encoder=_md5_key_encoder(settings.embedding_model),
        query_embedding_cache=True,  # 查询向量同样走缓存（与 Java 一致）
    )


def describe_embedding(settings: RagSettings) -> str:
    """生成嵌入模型配置的文字描述（供 CLI 展示）。"""
    key = settings.effective_embedding_api_key
    key_desc = key[:8] + "..." if key else "(未配置)"
    return (
        f"embedding_model={settings.embedding_model}"
        f" | base_url={settings.effective_embedding_base_url or '默认'}"
        f" | dimension={settings.embedding_dimension} | api_key={key_desc}"
    )
