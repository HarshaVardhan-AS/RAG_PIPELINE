import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Cleaned up imports - keeping them at the top
from app.models import GraphState
from app.services.vector_store import create_collection
from app.services.pdf_ingest import ingest_pdf
from app.services.tts import text_to_speech

print("RUNNING FILE:", __file__)
print("MAIN FILE LOADED — OPTIMIZED SINGLE-PASS PIPELINE")

os.makedirs("audio", exist_ok=True)
os.makedirs("data", exist_ok=True)

app = FastAPI(title="Agentic RAG API", version="1.0.0")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

class QueryRequest(BaseModel):
    query: str
    language: str = "en"

class QueryResponse(BaseModel):
    query: str
    answer: str
    audio_url: str
    documents: list

@app.get("/")
async def root():
    return {"message": "Agentic RAG Pipeline - Hackathon Template"}

@app.post("/query")
async def process_query(request: QueryRequest):
    # Importing here to avoid circular imports depending on your project structure
    from app.graph import graph

    try:
        # 🔥 OPTIMIZATION 1: No pre-translation. We pass the raw query and language directly to the graph.
        initial_state: GraphState = {
            "query": request.query,
            "language": request.language,  # Tell the graph what language to output
            "retrieved_documents": [],
            "relevance_score": None,
            "final_answer": None,
            "retry_count": 0
        }

        final_state = await graph.ainvoke(initial_state)

        # The answer coming back is ALREADY in the target language natively from Gemini
        answer = final_state.get("final_answer", "No answer generated")

        # 🔥 OPTIMIZATION 2: Pass the natively translated answer straight to TTS
        audio_file = text_to_speech(answer, request.language)
        audio_url = f"/{audio_file}"

        return {
            "query": request.query,
            "answer": answer,
            "audio_url": audio_url,
            "documents": final_state.get("retrieved_documents", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF into Qdrant/VectorDB
    """
    file_path = f"data/{file.filename}"

    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create / reset collection
    create_collection()

    # Ingest into vector DB
    ingest_pdf(file_path)

    return {
        "status": "success",
        "filename": file.filename
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}