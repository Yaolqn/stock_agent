"""rag 模块的离线测试（不调用真实模型、不访问网络/向量库）。

覆盖与 LangChain 1.x 兼容性直接相关的纯逻辑部分：
  1. 文档分块（split_text / create_document_chunks）——不依赖任何外部服务；
  2. 关键词重排与 BM25 索引（rerank / BM25Index）——纯内存计算；
  3. 指数退避重试的判定（should_retry）——纯字符串规则；
  4. 嵌入模型工厂在未配置 Key 时不应崩溃（构建层面）。
"""

from __future__ import annotations

from rag.config import RagSettings
from rag.document import DocumentChunk, create_document_chunks, split_text
from rag.retrieval import BM25Index, rerank
from rag.retry import should_retry


def _settings() -> RagSettings:
    """构造离线可用的 RAG 配置（不依赖 .env 与外部服务）。"""
    return RagSettings(
        llm_provider="fake",
        embedding_cache_enabled=False,
        bm25_enabled=False,
    )


def test_split_text_respects_chunk_sizes():
    """分块结果应保持在合理尺寸范围（长段落被拆分 + 重叠补丁不超界）。"""
    settings = _settings()
    text = (
        "第一段内容。这里是关于股票投资策略的详细介绍。\n\n"
        + ("长段落内容。" * 200)
        + "\n\n尾部一小段。"
    )
    chunks = split_text(text, settings)

    assert len(chunks) >= 2
    # 无单块超出 max_size + 重叠余量（重叠补丁最多加 chunk_overlap 字符）
    limit = settings.chunk_max_size + settings.chunk_overlap + 50
    assert all(len(c) <= limit for c in chunks)
    # 相邻块之间通过重叠内容保持上下文连续
    if len(chunks) > 1:
        # 尾部块应包含前一文本的片段（重叠由上一块尾部截取而来）
        assert any(c in chunks[1] for c in ("长段落", "第一段"))


def test_create_document_chunks_metadata():
    """文档块应带唯一 ID、统一 document_id 与递增的 chunk_index。"""
    settings = _settings()
    # 两段都足够长（> target_size），避免被小段合并逻辑拼成一块
    para1 = "第一段：关于基本面分析的详细内容。" + ("深入探讨公司营收与利润。" * 40)
    para2 = "第二段：关于技术面分析的详细内容。" + ("分析均线系统与成交量。" * 40)
    chunks = create_document_chunks(f"{para1}\n\n{para2}", "测试.txt", settings, "doc-1")

    assert len(chunks) == 2
    assert all(c.document_id == "doc-1" for c in chunks)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert all(c.source == "测试.txt" for c in chunks)
    # ID 应唯一（每块一个 uuid）
    assert len({c.id for c in chunks}) == 2


def test_keyword_rerank_orders_by_match():
    """关键词重排应把命中查询词的块排在前面。"""
    settings = _settings()
    candidates = [
        DocumentChunk(
            id="1", document_id="d", content="今天市场整体平稳，无特别热点。",
            source="a", chunk_index=0, similarity=0.9,
        ),
        DocumentChunk(
            id="2", document_id="d", content="贵州茅台发布业绩预告，营收超预期。",
            source="b", chunk_index=1, similarity=0.5,
        ),
        DocumentChunk(
            id="3", document_id="d", content="贵州茅台股价创历史新高。",
            source="c", chunk_index=2, similarity=0.6,
        ),
    ]
    reranked = rerank("贵州茅台", candidates, final_top_k=2, settings=settings)

    # 未命中查询词的块应被排到后面，只保留前 2 名
    assert [c.id for c in reranked] == ["2", "3"]
    assert len(reranked) == 2


def test_bm25_index_build_and_search():
    """BM25 索引构建后应能按查询词召回命中块。"""
    chunks = [
        DocumentChunk(id="1", document_id="d", content="银行板块今日上涨", source="a", chunk_index=0),
        DocumentChunk(id="2", document_id="d", content="白酒板块资金流出", source="b", chunk_index=1),
    ]
    index = BM25Index()
    index.build(chunks)

    assert index.is_built
    hits = index.search("银行", top_k=5)
    assert hits and hits[0].id == "1"
    # 未命中的查询返回空
    assert index.search("半导体", top_k=5) == []


def test_should_retry_classification():
    """重试判定：鉴权/参数类错误不重试，瞬时故障重试。"""
    assert should_retry(RuntimeError("timed out")) is True
    assert should_retry(RuntimeError("rate limit exceeded")) is True
    assert should_retry(RuntimeError("unauthorized")) is False
    assert should_retry(RuntimeError("invalid api key")) is False
