import asyncio
from google import genai
from app.config import settings


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = "gemini-2.5-flash"  # Use 2.0-flash if 2.5 is redlining

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async wrapper with built-in retries and fallback"""
        max_retries = 3
        retry_delay = 2  # seconds
        print("API KEY LOADED:", settings.google_api_key[:8])

        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": kwargs.get("temperature", 0.1),
                        "max_output_tokens": kwargs.get("max_tokens", 1024),
                    },
                )

                if hasattr(response, "text") and response.text:
                    return response.text.strip()

                # Fallback for complex response objects
                if hasattr(response, "candidates") and response.candidates:
                    content = response.candidates[0].content
                    return "".join([part.text for part in content.parts if hasattr(part, "text")]).strip()

            except Exception as e:
                # If we hit the 429 quota, wait and try again
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Quota hit. Retrying in {retry_delay}s... (Attempt {attempt + 1})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

                print(f"❌ Gemini API error: {e}")
                break

        # If all retries fail, return a safe string instead of crashing the server
        return "I'm sorry, I'm experiencing a high volume of requests. Please try again in a moment."


gemini_client = GeminiClient()