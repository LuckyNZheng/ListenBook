"""听书知识库主入口"""

import uvicorn

from listen_book.api.router import create_app


def main():
    """启动服务"""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
