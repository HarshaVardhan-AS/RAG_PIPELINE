from google import genai
import os
from app.config import settings


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(
            api_key=settings.google_api_key
        )
        self.model = "gemini-2.5-flash"

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async wrapper for Gemini API"""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_output_tokens": kwargs.get("max_tokens", 1024),
                },
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""


gemini_client = GeminiClient()