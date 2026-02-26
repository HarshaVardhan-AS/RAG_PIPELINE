from app.models import GraphState
from app.services.llm import gemini_client


async def rewrite_query_node(state: GraphState) -> GraphState:
    """Rewrite query for better retrieval"""
    query = state["query"]

    rewrite_prompt = f"""
    The following query did not retrieve relevant documents:
    "{query}"

    Rewrite this query to be more specific and retrieval-friendly.
    Return only the rewritten query.
    """

    rewritten_query = await gemini_client.generate_async(rewrite_prompt, temperature=0.3)

    state["query"] = rewritten_query.strip()
    state["retry_count"] = state.get("retry_count", 0) + 1

    print(f"🔄 Rewritten query: {state['query']}")

    return state
