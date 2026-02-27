from app.models import GraphState
from app.services.llm import groq_client


async def generator_node(state: GraphState) -> GraphState:
    """Generate final answer using retrieved documents"""
    query = state["query"]
    documents = state["retrieved_documents"]

    LANG_MAP = {
        "en": "English",
        "te": "Telugu",
        "hi": "Hindi",
        "ta": "Tamil"
    }

    target_language = LANG_MAP.get(state.get("language", "en"), "English")

    context = "\n\n".join([
        f"Document {i + 1}: {doc.get('content', '')}"
        for i, doc in enumerate(documents[:3])
    ])

    # HR and Company Policy focused prompt
    generation_prompt = f"""
You are a helpful HR and Company Policy assistant.

Your role is to help employees understand company policies, procedures, benefits, and workplace guidelines.

Answer using ONLY the provided context below.

Rules:
- Use simple, clear sentences
- No markdown formatting
- Plain text only (for text-to-speech)
- Be professional and helpful
- If the context doesn't contain the answer, say so politely
- Keep the answer concise and focused
- Maximum 5–6 sentences OR numbered points
- Include only the most relevant policy conditions
- Do NOT repeat unnecessary background text
- Use simple language for workers
- Output in the requested language
- Plain text only (no markdown symbols)

Language: Respond entirely in {target_language}.

Context:
{context}

Employee Question:
{query}

Your Answer (in {target_language}):
    """

    answer = await groq_client.generate_async(generation_prompt)

    state["final_answer"] = answer
    print(f"✨ Generated answer in {target_language}")

    return state