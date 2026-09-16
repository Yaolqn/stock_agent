"""文档处理模块：文本提取 + 语义分块（对齐 Java 参考项目 DocumentService）。

分块策略：
1. 先按空行切段落，保持段落完整性；
2. 段落过长（> max_size）时按句子边界拆分；
3. 段落过短（< min_size）的块后续与相邻块合并；
4. 合并到 target_size 附近，减少碎片；
5. 为相邻块添加 overlap 重叠文本，保持上下文连续。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import RagSettings

# 支持的文档格式
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_PDF_SUFFIXES = {".pdf"}
_DOCX_SUFFIXES = {".docx"}

# 段落分隔：连续空行
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# 句子分隔：中文/英文句末标点之后
_SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s*")


@dataclass
class DocumentChunk:
    """文档块：内容 + 元数据 + 各种检索分数（对齐 Java 的 DocumentChunk）。"""

    id: str
    document_id: str
    content: str
    source: str
    chunk_index: int
    total_chunks: int = 0
    similarity: float = 0.0   # 向量相似度
    score: float = 0.0        # BM25 分数
    hybrid_score: float = 0.0  # 混合检索分数
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 文本提取
# --------------------------------------------------------------------------- #
def _read_text_file(path: Path) -> str:
    """读取纯文本文件，自动尝试常见编码。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 兜底：忽略无法解码的字符
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_file(path: Path) -> str:
    """提取 PDF 文本（逐页拼接）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _read_docx_file(path: Path) -> str:
    """提取 Word(.docx) 文本（段落拼接）。"""
    import docx

    document = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def extract_text(path: str | Path) -> str:
    """从文件中提取纯文本。

    Args:
        path: 文件路径。

    Returns:
        提取出的文本内容。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不支持的文件格式。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix in _PDF_SUFFIXES:
        text = _read_pdf_file(file_path)
    elif suffix in _DOCX_SUFFIXES:
        text = _read_docx_file(file_path)
    elif suffix in _TEXT_SUFFIXES:
        text = _read_text_file(file_path)
    else:
        raise ValueError(
            f"不支持的文件格式：{suffix or '(无扩展名)'}，"
            "当前支持 PDF / DOCX / TXT / Markdown"
        )

    if not text.strip():
        raise ValueError(f"未能从文件中提取到文本：{file_path.name}")
    return text


# --------------------------------------------------------------------------- #
# 语义分块
# --------------------------------------------------------------------------- #
def _split_long_paragraph(paragraph: str, settings: RagSettings) -> list[str]:
    """按句子边界拆分过长的段落。"""
    chunks: list[str] = []
    current = ""

    for sentence in _SENTENCE_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) > settings.chunk_max_size and \
                len(current) >= settings.chunk_min_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence

    if current:
        chunks.append(current.strip())
    return chunks


def _merge_small_chunks(chunks: list[str], settings: RagSettings) -> list[str]:
    """把过短的块合并到目标大小附近。"""
    merged: list[str] = []
    current = ""

    for chunk in chunks:
        if len(current) + len(chunk) <= settings.chunk_target_size:
            current = f"{current}\n\n{chunk}" if current else chunk
        else:
            if current:
                merged.append(current)
            current = chunk

    if current:
        merged.append(current)
    return merged


def _get_overlap_text(text: str, overlap_size: int) -> str:
    """截取文本尾部的重叠部分（尽量从句子边界开始）。"""
    if len(text) <= overlap_size:
        return text

    tail = text[len(text) - overlap_size:]
    first_sentence_end = tail.find("。")
    if first_sentence_end > 0:
        return tail[first_sentence_end + 1:].strip()
    return tail.strip()


def _add_overlap(chunks: list[str], settings: RagSettings) -> list[str]:
    """为每个块前置上一块尾部的重叠内容。"""
    if len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            overlap = _get_overlap_text(chunks[i - 1], settings.chunk_overlap)
            chunk = f"{overlap}\n\n{chunk}"
        overlapped.append(chunk)
    return overlapped


def split_text(text: str, settings: RagSettings) -> list[str]:
    """把整篇文本切分为带重叠的语义块。"""
    chunks: list[str] = []

    for paragraph in _PARAGRAPH_RE.split(text):
        if not paragraph.strip():
            continue

        length = len(paragraph)
        if settings.chunk_min_size <= length <= settings.chunk_max_size:
            chunks.append(paragraph.strip())
            continue

        if length > settings.chunk_max_size:
            chunks.extend(_split_long_paragraph(paragraph, settings))
        elif length < settings.chunk_min_size:
            chunks.append(paragraph.strip())

    chunks = _merge_small_chunks(chunks, settings)
    return _add_overlap(chunks, settings)


def create_document_chunks(
    text: str,
    source: str,
    settings: RagSettings,
    document_id: str | None = None,
) -> list[DocumentChunk]:
    """把文本切分为文档块对象列表（每块带唯一 ID 与所属文档 ID）。"""
    doc_id = document_id or str(uuid.uuid4())
    chunks = split_text(text, settings)
    total = len(chunks)

    return [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            content=content,
            source=source,
            chunk_index=index,
            total_chunks=total,
        )
        for index, content in enumerate(chunks)
    ]
