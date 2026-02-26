import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from google import genai

load_dotenv()

COLLECTION_NAME = "hackathon"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def create_collection():
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,
            distance=Distance.COSINE,
        ),
    )


def embed_text(text: str):
    emb = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return emb.embeddings[0].values


def search(query: str, limit: int = 3):
    vector = embed_text(query)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
    )

    return [r.payload["text"] for r in results]