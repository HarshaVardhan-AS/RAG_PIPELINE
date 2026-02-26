from langgraph.graph import StateGraph, END
from app.models import GraphState
from app.nodes.retriever import retriever_node
from app.nodes.grader import grader_node
from app.nodes.generator import generator_node
from app.nodes.rewrite import rewrite_query_node
from app.config import settings


def should_continue(state: GraphState) -> str:
    """Routing logic after grading"""
    relevance_score = state.get("relevance_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if retry_count >= settings.max_retries:
        print(f"⚠️ Max retries reached, generating anyway")
        return "generate"

    if relevance_score >= settings.relevance_threshold:
        return "generate"
    else:
        return "rewrite"


def create_graph() -> StateGraph:
    """Build the agentic RAG graph"""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("grade", grader_node)
    workflow.add_node("generate", generator_node)
    workflow.add_node("rewrite", rewrite_query_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        should_continue,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "retrieve")

    return workflow.compile()


graph = create_graph()
