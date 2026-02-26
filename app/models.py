from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    """State for the RAG graph"""
    query: str
    retrieved_documents: List[dict]
    relevance_score: Optional[float]
    final_answer: Optional[str]
    retry_count: int
