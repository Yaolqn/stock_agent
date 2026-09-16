"""rag —— 终端入口（基于文档的检索增强问答）。

用法：
    python -m rag.main
    .\\.venv\\Scripts\\python.exe -m rag.main

流程与 Java 参考项目一致：
    上传文档 → 提取文本 → 语义分块 → 生成嵌入（可走 Redis 缓存）
    → 写入 Milvus → 检索相关块（可选重排 / BM25 混合）→ 拼接上下文 → 聊天模型作答（带溯源）

聊天模型复用本项目 app.models.create_llm；向量库复用 Milvus；缓存复用 Redis。
用 `/help` 查看命令；用 `/exit` 或按 Ctrl+C 退出。
"""

from __future__ import annotations

import sys

from app.models import describe_llm
from rag.config import get_rag_settings
from rag.embeddings import describe_embedding
from rag.service import RagService
from rag.vectorstore import describe_vector_store

BANNER = f"""
{'=' * 58}
  文档问答 RAG（rag）
  · 上传文档 → 向量化入库（Milvus）→ 检索增强问答
  · 支持 PDF / Word(.docx) / TXT / Markdown
  输入 /help 查看命令；输入 /exit 或按 Ctrl+C 退出
{'=' * 58}
"""

HELP_TEXT = """可用命令：
  /upload <文件路径>   上传并入库一个文档
                         例：/upload D:\\docs\\产品手册.pdf
  /chat <问题>         基于已入库文档提问（也可直接输入问题）
  /select <文档ID|all> 限定后续提问的检索范围（默认 all=全库）
  /documents           列出已入库的文档（ID / 文件名 / 分块数）
  /status              查看向量库状态（分块总数、文档数量）
  /delete <文档ID>     删除指定文档的全部分块
  /clear               清空整个向量库（危险操作）
  /model               查看当前聊天模型 / 嵌入模型 / 向量库配置
  /help                显示本帮助
  /exit                退出程序（或按 Ctrl+C / Ctrl+D）
"""


def _ensure_utf8_stdio() -> None:
    """确保 Windows 终端能正确显示中文。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _build_service() -> RagService | None:
    """创建 RAG 服务；向量库不可用时给出明确指引。"""
    try:
        return RagService(get_rag_settings())
    except Exception as exc:  # noqa: BLE001 —— 初始化失败需要给出可操作提示
        print(f"\n[环境错误] 初始化 RAG 服务失败：{exc}")
        print("  请确认 Milvus 已启动且 MILVUS_HOST / MILVUS_PORT 配置正确：")
        print("    docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest")
        print("  向量库连接地址可在 .env 中查看（MILVUS_HOST / MILVUS_PORT）。")
        return None


def _print_retrieved(service: RagService) -> None:
    """打印本次问答命中的来源（便于核对答案依据）。"""
    chunks = service.last_retrieved
    if not chunks:
        print("  [检索] 未命中任何文档块。")
        return
    sources = list(dict.fromkeys(chunk.source or "(未知来源)" for chunk in chunks))
    print(f"  [检索] 命中 {len(chunks)} 个文档块，来源：{', '.join(sources)}")


def _handle_upload(service: RagService, argument: str) -> None:
    """处理 /upload 命令。"""
    if not argument:
        print("[引导] /upload 后面需要带上文件路径，例如：\n  /upload D:\\docs\\产品手册.pdf")
        return
    print(f"\n[上传] 开始处理：{argument}")
    try:
        result = service.upload(argument)
    except Exception as exc:  # noqa: BLE001 —— 上传失败给出原因而非中断程序
        print(f"[错误] 上传失败：{exc}")
        return
    print(f"[上传] 完成：{result.filename}")
    print(f"  文档ID：{result.document_id}")
    print(f"  分块数：{result.chunks}（库内总量：{result.total_chunks}）")


def _handle_ask(service: RagService, question: str, document_id: str | None) -> None:
    """处理一次 RAG 问答（流式输出）。"""
    if not question:
        print("[引导] 请直接输入问题，或使用 /chat 你的问题")
        return
    scope = document_id or "全库"
    print(f"[检索] 范围：{scope}")
    print("助手 > ", end="", flush=True)
    collected: list[str] = []
    for piece in service.stream_chat(question, document_id):
        print(piece, end="", flush=True)
        collected.append(piece)
    print()
    if not collected:
        print("（模型未返回可显示内容）")
    _print_retrieved(service)


def main() -> None:
    """rag 终端入口：上传文档 → 向量化入库 → 检索增强问答。

    流程：初始化 RAG 服务 → 打印模型/嵌入/向量库配置 → 进入交互循环。
    /upload 入库、/chat 或直接输入提问、/documents 等命令管理向量库。
    """
    _ensure_utf8_stdio()
    settings = get_rag_settings()

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print("[提示] 未检测到 OPENAI_API_KEY，真实问答将报错；")
        print("       请先在 .env 中配置 Key，或用 LLM_PROVIDER=fake 体验演示模式。")
    if not settings.effective_embedding_api_key:
        print("[提示] 未检测到 EMBEDDING_API_KEY（也未回退到 OPENAI_API_KEY），文档向量化会失败。")

    print(BANNER)
    print(HELP_TEXT)

    service = _build_service()
    if service is None:
        return

    print(f"[模型] {describe_llm(settings)}")
    print(f"[嵌入] {describe_embedding(settings)}")
    print(f"[向量库] {describe_vector_store(settings)}")
    print(f"[重排] 启用={settings.rerank_enabled} | 方式={settings.rerank_method} "
          f"| BM25={settings.bm25_enabled}(权重={settings.bm25_weight})")

    active_document_id: str | None = None

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            low = user_input.lower()
            if low in ("/exit", "/quit"):
                print("再见！")
                break
            elif low == "/help":
                print(HELP_TEXT)
            elif low.startswith("/upload"):
                _handle_upload(service, user_input[len("/upload"):].strip())
            elif low.startswith("/chat"):
                _handle_ask(
                    service,
                    user_input[len("/chat"):].strip(),
                    active_document_id,
                )
            elif low.startswith("/select"):
                argument = user_input[len("/select"):].strip()
                if not argument:
                    print(f"当前检索范围：{active_document_id or '全库'}")
                elif argument.lower() == "all":
                    active_document_id = None
                    print("已将检索范围设为全库。")
                else:
                    active_document_id = argument
                    print(f"已将检索范围设为文档：{argument}")
            elif low == "/documents":
                documents = service.list_documents()
                if not documents:
                    print("（向量库为空，请先用 /upload 上传文档）")
                else:
                    print(f"共 {len(documents)} 个文档：")
                    for doc_id, filename, count in documents:
                        print(f"  - {doc_id}  {filename}  分块数={count}")
            elif low == "/status":
                status = service.status()
                print(f"向量库分块总数：{status['total_chunks']}")
                print(f"文档数量：{len(status['documents'])}")
            elif low.startswith("/delete"):
                argument = user_input[len("/delete"):].strip()
                if not argument:
                    print("[引导] /delete 后面需要带上文档ID（可用 /documents 查看）")
                else:
                    service.delete_document(argument)
                    if active_document_id == argument:
                        active_document_id = None
                    print(f"已删除文档：{argument}")
            elif low == "/clear":
                confirm = input("确认清空整个向量库？该操作不可恢复 (y/N) > ").strip().lower()
                if confirm == "y":
                    service.clear()
                    active_document_id = None
                    print("向量库已清空。")
                else:
                    print("已取消。")
            elif low == "/model":
                print(f"[聊天] {describe_llm(settings)}")
                print(f"[嵌入] {describe_embedding(settings)}")
                print(f"[向量库] {describe_vector_store(settings)}")
            else:
                print(f"未知命令：{user_input}，输入 /help 查看帮助。")
            continue

        # 普通输入：直接走 RAG 问答
        try:
            _handle_ask(service, user_input, active_document_id)
        except Exception as exc:  # noqa: BLE001 —— 交互程序需要兜底所有异常
            print(f"\n[错误] {exc}")
            print("[提示] 请检查 API Key / Base URL / 网络（代理）配置；输入 /model 查看当前配置。")


if __name__ == "__main__":
    main()
