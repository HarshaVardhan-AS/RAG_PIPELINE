from app.models import GraphState
from app.services.vector_store import search


async def retriever_node(state: GraphState) -> GraphState:
    query = state["query"]

    docs = search(query)

    state["retrieved_documents"] = [
        {"content": d} for d in docs
    ]

    print(f"✅ Retrieved {len(docs)} documents")

    return state