from app.models import GraphState
from app.services.llm import gemini_client


async def generator_node(state: GraphState) -> GraphState:
    """Generate final answer using retrieved documents"""
    query = state["query"]
    documents = state["retrieved_documents"]

    context = "\n\n".join([
        f"Document {i+1}: {doc.get('content', '')}"
        for i, doc in enumerate(documents[:3])
    ])

    generation_prompt = f"""
    Based on the following context, answer the question.

    Context:
    {context}

    Question: {query}

    Answer:
    """

    answer = await gemini_client.generate_async(generation_prompt)

    state["final_answer"] = answer
    print(f"✨ Generated answer")

    return state
