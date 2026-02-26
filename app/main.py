from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.models import GraphState
print("RUNNING FILE:", __file__)
import shutil
import os
from fastapi import UploadFile, File

from app.services.vector_store import create_collection
from app.services.pdf_ingest import ingest_pdf
print("MAIN FILE LOADED — WITH UPLOAD ENDPOINT")


app = FastAPI(title="Agentic RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: list


@app.get("/")
async def root():
    return {"message": "Agentic RAG Pipeline - Hackathon Template"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main RAG endpoint"""
    from app.graph import graph
    try:
        initial_state: GraphState = {
            "query": request.query,
            "retrieved_documents": [],
            "relevance_score": None,
            "final_answer": None,
            "retry_count": 0
        }
        final_state = await graph.ainvoke(initial_state)
        return QueryResponse(
            query=final_state["query"],
            answer=final_state.get("final_answer", "No answer generated"),
            documents=final_state.get("retrieved_documents", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")






@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF into Qdrant
    """

    os.makedirs("data", exist_ok=True)
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
