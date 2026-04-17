"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from listen_book.processor.query_processor.state import QueryGraphState
from listen_book.processor.query_processor.nodes.book_confirmed_node import BookConfirmedNode
from listen_book.processor.query_processor.nodes.query_rewrite_node import QueryRewriteNode
from listen_book.processor.query_processor.nodes.book_validation_node import BookValidationNode
from listen_book.processor.query_processor.nodes.hybrid_vector_search_node import HybridVectorSearchNode
from listen_book.processor.query_processor.nodes.hyde_vector_search_node import HydeVectorSearchNode
from listen_book.processor.query_processor.nodes.rrf_merge_node import RrfMergeNode
from listen_book.processor.query_processor.nodes.reranker_node import RerankerNode
from listen_book.processor.query_processor.nodes.answer_output_node import AnswerOutputNode


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图

    流程结构::

        book_confirmed_node
              │
              v
        query_rewrite (利用历史上下文重写查询)
              │
              v
        book_validation (验证书名并对齐知识库)
              │
              v
        multi_search (虚拟节点，分发)
              │
        ┌─────┼─────┐
        │     │     │
        v     v     v
      dense  hyde  (预留web搜索)
        │     │     │
        └─────┴─────┘
              │
              v
           rrf_merge
              │
              v
         reranker
              │
              v
      answer_output
              │
              v
            END
    """
    workflow = StateGraph(QueryGraphState)  # type: ignore

    # 定义节点
    nodes = {
        "book_confirmed_node": BookConfirmedNode(),
        "query_rewrite_node": QueryRewriteNode(),
        "book_validation_node": BookValidationNode(),
        "multi_search": lambda x: {},  # 虚拟分发节���
        "hybrid_vector_search_node": HybridVectorSearchNode(),
        "hyde_vector_search_node": HydeVectorSearchNode(),
        "join": lambda x: {},  # 虚拟汇合节点
        "rrf_merge_node": RrfMergeNode(),
        "reranker_node": RerankerNode(),
        "answer_output_node": AnswerOutputNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)  # type: ignore

    # 入口点
    workflow.set_entry_point("book_confirmed_node")

    # 顺序边：书名确认 -> 查询重写 -> 书名验证 -> 分发
    workflow.add_edge("book_confirmed_node", "query_rewrite_node")
    workflow.add_edge("query_rewrite_node", "book_validation_node")
    workflow.add_edge("book_validation_node", "multi_search")

    # 多路并行分发
    workflow.add_edge("multi_search", "hybrid_vector_search_node")
    workflow.add_edge("multi_search", "hyde_vector_search_node")

    # 多路汇合
    workflow.add_edge("hybrid_vector_search_node", "join")
    workflow.add_edge("hyde_vector_search_node", "join")

    # RRF 融合 -> Rerank -> 答案生成
    workflow.add_edge("join", "rrf_merge_node")
    workflow.add_edge("rrf_merge_node", "reranker_node")
    workflow.add_edge("reranker_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


# 全局实例
query_app = create_query_graph()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    test_state = {
        "session_id": "test_session",
        "task_id": "test_001",
        "original_query": "《三体》这本书讲什么？",
        "is_stream": False,
    }

    result = query_app.invoke(test_state)
    print(f"答案: {result.get('answer', '')}")