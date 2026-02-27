import asyncio
from groq import AsyncGroq
from app.config import settings


class GroqClient:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Using Llama 3.3 70B - one of the best models on Groq
        self.model = "llama-3.3-70b-versatile"
        # Alternative models you can use:
        # "llama-3.1-70b-versatile" - Fast and capable
        # "mixtral-8x7b-32768" - Good for longer context
        # "gemma2-9b-it" - Smaller, faster model

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async wrapper with built-in retries and fallback"""
        max_retries = 3
        retry_delay = 2  # seconds
        print("GROQ API KEY LOADED:", settings.groq_api_key[:8] if settings.groq_api_key else "MISSING")

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 1500),
                )

                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()

            except Exception as e:
                # If we hit rate limits, wait and try again
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"⚠️ Rate limit hit. Retrying in {retry_delay}s... (Attempt {attempt + 1})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

                print(f"❌ Groq API error: {e}")
                break

        # If all retries fail, return a safe string instead of crashing the server
        return "I'm sorry, I'm experiencing a high volume of requests. Please try again in a moment."


groq_client = GroqClient()

# Keep backward compatibility alias
llm_client = groq_client
