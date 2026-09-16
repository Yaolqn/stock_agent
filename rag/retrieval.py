"""检索服务：向量检索 + 关键词重排 + BM25 混合检索。

对齐 Java 参考项目的 RetrievalService / RerankService / BM25Service：
- 向量检索：Milvus 相似度召回（支持文档隔离）；
- 重排：keyword（关键词出现次数 + 位置权重）或 semantic（70% 向量 + 30% 关键词）；
- BM25 混合检索：向量分数与 BM25 分数分别归一化后按权重相加。
"""

from __future__ import annotations

import re
from dataclasses import replace

from rag.config import RagSettings
from rag.document import DocumentChunk
from rag.vectorstore import MilvusVectorStore

# 中文停用词（与 Java 端一致）
_STOP_WORDS = {
    "的", "了", "是", "在", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看",
    "好", "自己", "这",
}

# 关键词切分：空格与常见中英文标点
_SPLIT_RE = re.compile(r"[\s,，.。!！?？;；:：]")
# 分词：连续英文数字视为一个词，中文按单字切分（近似 Lucene StandardAnalyzer）
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _extract_keywords(query: str) -> set[str]:
    """提取查询关键词（去停用词、去单字、统一小写）。"""
    keywords = set()
    for word in _SPLIT_RE.split(query):
        word = word.strip()
        if len(word) > 1 and word not in _STOP_WORDS:
            keywords.add(word.lower())
    return keywords


def _count_occurrences(text: str, substring: str) -> int:
    """统计子串出现次数（非重叠）。"""
    count = 0
    index = text.find(substring)
    while index != -1:
        count += 1
        index = text.find(substring, index + len(substring))
    return count


def _keyword_score(content: str, keywords: set[str]) -> float:
    """关键词匹配打分：出现次数 × (1 + 位置权重)，再按关键词数归一化。"""
    if not keywords:
        return 0.0

    lower_content = content.lower()
    score = 0.0
    total_matches = 0

    for keyword in keywords:
        if keyword in lower_content:
            total_matches += 1
            count = _count_occurrences(lower_content, keyword)
            first_index = lower_content.find(keyword)
            position_weight = 1.0 - first_index / len(lower_content)
            score += count * (1 + position_weight)

    return score / len(keywords) if total_matches > 0 else score


def _keyword_rerank(query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
    """基于关键词匹配重排（用关键词分覆盖 similarity）。"""
    keywords = _extract_keywords(query)
    for chunk in candidates:
        chunk.similarity = _keyword_score(chunk.content, keywords)
    return sorted(candidates, key=lambda c: c.similarity, reverse=True)


def _semantic_rerank(query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
    """基于语义相似度重排（70% 原始向量相似度 + 30% 关键词分）。"""
    keywords = _extract_keywords(query)
    for chunk in candidates:
        original = chunk.similarity
        chunk.similarity = 0.7 * original + 0.3 * _keyword_score(chunk.content, keywords)
    return sorted(candidates, key=lambda c: c.similarity, reverse=True)


def rerank(
    query: str,
    candidates: list[DocumentChunk],
    final_top_k: int,
    settings: RagSettings,
) -> list[DocumentChunk]:
    """对候选块重排并截断为 final_top_k 个。"""
    if not settings.rerank_enabled or not candidates:
        return candidates
    if len(candidates) <= final_top_k:
        return candidates

    if settings.rerank_method.lower() == "semantic":
        reranked = _semantic_rerank(query, candidates)
    else:
        reranked = _keyword_rerank(query, candidates)
    return reranked[:final_top_k]


class BM25Index:
    """基于 rank_bm25 的关键词索引（对齐 Java 的 Lucene 内存索引）。"""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25 = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        """用给定文档块重建索引。"""
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        if not self._chunks:
            self._corpus_tokens = []
            self._bm25 = None
            return

        corpus = [_TOKEN_RE.findall(chunk.content.lower()) for chunk in self._chunks]
        # 避免整篇无有效 token 导致 BM25 统计异常
        corpus = [tokens or [""] for tokens in corpus]
        self._corpus_tokens = corpus
        self._bm25 = BM25Okapi(corpus)

    def clear(self) -> None:
        """清空索引（下次检索时会重新构建）。"""
        self._chunks = []
        self._corpus_tokens = []
        self._bm25 = None

    @property
    def is_built(self) -> bool:
        """索引是否已构建（未构建时检索会先全量加载）。"""
        return self._bm25 is not None

    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[DocumentChunk]:
        """BM25 检索（可选文档隔离），按分数降序返回。"""
        if self._bm25 is None:
            return []

        query_tokens = _TOKEN_RE.findall(query.lower())
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)

        results: list[DocumentChunk] = []
        for index in ranked:
            chunk = self._chunks[index]
            if document_id and chunk.document_id != document_id:
                continue
            # 与 Lucene 语义对齐：只返回至少命中一个查询词的块。
            # （不能按 score>0 过滤：当某词出现在过半文档中时 BM25 的 idf 为负，
            #   真正命中的块分数可能为负，会被误丢弃。）
            if not query_token_set.intersection(self._corpus_tokens[index]):
                continue
            results.append(replace(chunk, score=float(scores[index])))
            if len(results) >= top_k:
                break
        return results


class RetrievalService:
    """检索编排：向量检索（默认）或向量 + BM25 混合检索。"""

    def __init__(self, settings: RagSettings, vector_store: MilvusVectorStore) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.bm25 = BM25Index()

    def invalidate_bm25(self) -> None:
        """文档发生增删后调用，使 BM25 索引在下次检索时重建。"""
        self.bm25.clear()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[DocumentChunk]:
        """检索与查询相关的文档块。"""
        k = top_k or self.settings.retrieval_top_k
        if self.settings.bm25_enabled:
            return self._hybrid_retrieve(query, k, document_id)
        return self._vector_retrieve(query, k, document_id)

    def _vector_retrieve(
        self, query: str, top_k: int, document_id: str | None
    ) -> list[DocumentChunk]:
        """纯向量检索（可选重排）。"""
        s = self.settings
        candidate_top_k = s.rerank_top_k if s.rerank_enabled else top_k
        candidates = self.vector_store.similarity_search(query, candidate_top_k, document_id)

        if s.rerank_enabled and len(candidates) > top_k:
            print(f"  启用重排序，候选块数量: {len(candidates)}，最终返回: {top_k}")
            return rerank(query, candidates, top_k, s)
        return candidates[:top_k]

    def _hybrid_retrieve(
        self, query: str, top_k: int, document_id: str | None
    ) -> list[DocumentChunk]:
        """向量 + BM25 混合检索。"""
        s = self.settings

        # 1. 向量召回
        vector_results = self.vector_store.similarity_search(query, s.bm25_top_k, document_id)

        # 2. BM25 召回（索引未构建时先从向量库全量加载）
        if not self.bm25.is_built:
            self.bm25.build(self.vector_store.get_all_chunks())
        bm25_results = self.bm25.search(query, s.bm25_top_k, document_id)

        # 3. 归一化后加权合并（同 ID 分数累加）
        merged: dict[str, DocumentChunk] = {}
        max_vector_score = max((c.similarity for c in vector_results), default=1.0) or 1.0
        for chunk in vector_results:
            chunk.hybrid_score = (chunk.similarity / max_vector_score) * (1 - s.bm25_weight)
            merged[chunk.id] = chunk

        max_bm25_score = max((c.score for c in bm25_results), default=1.0) or 1.0
        for chunk in bm25_results:
            contribution = (chunk.score / max_bm25_score) * s.bm25_weight
            existing = merged.get(chunk.id)
            if existing is not None:
                existing.hybrid_score += contribution
            else:
                chunk.hybrid_score = contribution
                merged[chunk.id] = chunk

        # 4. 按混合分数降序取 top_k
        return sorted(merged.values(), key=lambda c: c.hybrid_score, reverse=True)[:top_k]


def format_context(chunks: list[DocumentChunk]) -> str:
    """把检索结果格式化为带来源标注的上下文文本。"""
    return "\n\n".join(f"[来源: {chunk.source}]\n{chunk.content}" for chunk in chunks)
