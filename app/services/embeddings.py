import google.generativeai as genai
from typing import List
from app.config import settings


class EmbeddingService:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model = "models/gemini-embedding-001"

    async def embed_query(self, text: str) -> List[float]:
        """TODO: Replace with actual embedding logic"""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768


embedding_service = EmbeddingService()
