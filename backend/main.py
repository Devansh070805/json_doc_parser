from fastapi import FastAPI, Header, HTTPException
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, HttpUrl
from typing import List
import hashlib

# Import config and services
from core.config import key
from services import document_service
from services import pinecone_service  # ✅ replaced faiss_service with pinecone_service
from services.openrouter_service import answer_question_with_context  # ✅ imported function

# Security scheme
security = HTTPBearer()

# Cache to avoid reprocessing (only for avoiding repeated document processing)
document_cache = {}

# Request/Response models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="A system to process documents and answer questions using AI + Pinecone.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Server running successfully."}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(
    request: QueryRequest,
    authorization: str = Header(...)
):
    # --- 1. Auth check ---
    token = authorization.split(" ")[1] if " " in authorization else None
    if not token or token != key.hackrx_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # --- 2. Hash doc to cache ---
    doc_url = str(request.documents)
    doc_hash = hashlib.md5(doc_url.encode()).hexdigest()

    if doc_hash in document_cache:
        chunks = document_cache[doc_hash]
    else:
        try:
            document_text = document_service.extract_text_from_url(doc_url)
            chunks = document_service.chunk_text(document_text)
            pinecone_service.embed_and_store_chunks(chunks, doc_hash)  # ✅ store in Pinecone
            document_cache[doc_hash] = chunks
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Document processing failed: {str(e)}")

    # --- 3. Answer questions ---
    answers = []
    for question in request.questions:
        try:
            relevant_chunks = pinecone_service.search_chunks(question, k=5)  # ✅ search in Pinecone
            context = "\n".join(relevant_chunks)
            answer = answer_question_with_context(context, question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {question}. Error: {str(e)}")

    return QueryResponse(answers=answers)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
