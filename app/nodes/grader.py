from app.models import GraphState
from app.services.llm import groq_client


async def grader_node(state: GraphState) -> GraphState:
    """Grade relevance of retrieved documents"""
    query = state["query"]
    documents = state["retrieved_documents"]

    if not documents:
        state["relevance_score"] = 0.0
        return state

    grading_prompt = f"""
    Query: {query}

    Documents: {documents[:3]}

    Are these documents relevant to answer the query?
    Respond with only 'yes' or 'no'.
    """

    response = await groq_client.generate_async(grading_prompt, temperature=0.1)

    is_relevant = "yes" in response.lower()
    state["relevance_score"] = 1.0 if is_relevant else 0.0

    print(f"📊 Relevance score: {state['relevance_score']}")

    return state
