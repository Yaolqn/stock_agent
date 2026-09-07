"""LangChain AI 聊天助手 —— 程序入口。

用法：
    python main.py                  # 直接运行（配置见 .env）
    pip install -e . && ai-chat     # 安装为系统命令后运行
"""

from app.cli import main

if __name__ == "__main__":
    main()
