from langgraph.graph import StateGraph, END
from app.models import GraphState

from app.nodes.retriever import retriever_node
from app.nodes.generator import generator_node

def create_graph() -> StateGraph:
    """Build the FAST, open-loop RAG graph for the hackathon"""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", generator_node)

    # Straight line from start to finish. No loops.
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

graph = create_graph()