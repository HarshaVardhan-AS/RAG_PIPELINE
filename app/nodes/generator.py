from app.models import GraphState
from app.services.llm import gemini_client


async def generator_node(state: GraphState) -> GraphState:
    """Generate final answer using retrieved documents"""
    query = state["query"]
    documents = state["retrieved_documents"]

    # 👈 OPTIMIZATION: Pull the target language from the state
    target_language = state.get("language", "en")

    context = "\n\n".join([
        f"Document {i + 1}: {doc.get('content', '')}"
        for i, doc in enumerate(documents[:3])
    ])

    # 👈 Swapped the medical jargon for HR policy rules and added the translation instruction
    generation_prompt = f"""
    You are a helpful HR and Company Policy training assistant.

    Using ONLY the provided context, answer the employee's question clearly.

    CRITICAL INSTRUCTION: You MUST output your final answer entirely and fluently in the language corresponding to this language code: {target_language}. Do not use English unless the code is 'en'.

    Rules:
    - Do NOT summarize excessively
    - Do NOT merge different policies into one line
    - Explain each relevant rule or policy separately
    - Preserve all important information from the context
    - Use simple sentences (strictly NO markdown or asterisks, plain text only for text-to-speech)

    Format your response cleanly. 
    If multiple rules apply, list them clearly in plain text.

    Context:
    {context}

    Question: {query}

    Answer (in {target_language}):
    """

    answer = await gemini_client.generate_async(generation_prompt)

    state["final_answer"] = answer
    print(f"✨ Generated answer in {target_language}")

    return state