from pypdf import PdfReader
from qdrant_client.models import PointStruct
import uuid

from app.services.vector_store import qdrant, embed_text, COLLECTION_NAME


def load_pdf_text(path: str):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text_by_disease(text: str):
    sections = text.split("Disease:")
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        chunks.append("Disease: " + sec)

    return chunks


def ingest_pdf(path: str):
    text = load_pdf_text(path)
    chunks = chunk_text_by_disease(text)

    points = []

    for chunk in chunks:
        vector = embed_text(chunk)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk},
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"✅ Uploaded {len(points)} chunks")