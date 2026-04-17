"""导入流程主图

使用 LangGraph 构建文档导入工作流。
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.processor.import_processor.nodes.entry_node import EntryNode
from listen_book.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from listen_book.processor.import_processor.nodes.book_recognition_node import BookRecognitionNode
from listen_book.processor.import_processor.nodes.book_info_enrich_node import BookInfoEnrichNode
from listen_book.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from listen_book.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode


def import_router(state: ImportGraphState) -> str:
    """根据文件类型路由"""
    if state.get("is_pdf_enabled"):
        # PDF 需要先转换（暂不支持）
        return END
    if state.get("is_md_enabled"):
        return "document_split_node"
    return END


def import_graph() -> CompiledStateGraph:
    """创建导入流程图

    流程结构::

        entry_node
              │
              └── (MD) ──> document_split_node ──> book_recognition_node
                                                      │
                                                      v
                                              book_info_enrich_node (LLM 补充信息)
                                                      │
                                                      v
                                              embedding_chunks_node
                                                      │
                                                      v
                                              import_milvus_node
                                                      │
                                                      v
                                                    END
    """
    workflow = StateGraph(ImportGraphState)  # type: ignore

    # 定义节点
    nodes = {
        "entry_node": EntryNode(),
        "document_split_node": DocumentSplitNode(),
        "book_recognition_node": BookRecognitionNode(),
        "book_info_enrich_node": BookInfoEnrichNode(),
        "embedding_chunks_node": EmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)  # type: ignore

    # 入口点
    workflow.set_entry_point("entry_node")

    # 条件边：根据文件类型路由
    workflow.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "document_split_node": "document_split_node",
            END: END,
        },
    )

    # 顺序边：书名识别 -> 信息补充 -> 向量生成
    workflow.add_edge("document_split_node", "book_recognition_node")
    workflow.add_edge("book_recognition_node", "book_info_enrich_node")
    workflow.add_edge("book_info_enrich_node", "embedding_chunks_node")
    workflow.add_edge("embedding_chunks_node", "import_milvus_node")
    workflow.add_edge("import_milvus_node", END)

    return workflow.compile()


# 全局实例
import_app = import_graph()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    test_state = {
        "task_id": "test_001",
        "file_path": "/path/to/test.md",
    }

    for event in import_app.stream(test_state):
        for key, value in event.items():
            print(f"节点: {key}")
            print(f"状态: {value}")