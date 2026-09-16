"""rag —— 基于文档检索增强生成（RAG）的问答子系统。

仿造 Java 参考项目（C:\\Users\\80946\\Desktop\\JAVA\\RAG）实现，能力对齐：
    config.py       配置中心（在既有聊天配置之上追加 RAG 配置）
    retry.py        指数退避重试（对齐 RetryUtil）
    embeddings.py   嵌入模型工厂 + Redis 缓存（缓存键与 Java 端互通）
    document.py     文档提取与语义分块（对齐 DocumentService）
    vectorstore.py  Milvus 向量库封装（对齐 VectorStoreService）
    retrieval.py    向量检索 + 关键词重排 + BM25 混合检索
    prompts.py      RAG 提示词模板（对齐 RagService 内联提示词）
    service.py      RAG 问答服务（对齐 RagService / RagController）
    main.py         终端 CLI 入口（python -m rag.main）

复用点：聊天模型复用 app.models.create_llm；向量库复用 Java 端的 Milvus；
缓存复用 Java 端的 Redis。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
