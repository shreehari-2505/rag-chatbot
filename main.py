from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from document_store import DocumentStore
from settings import settings
import uvicorn

store = DocumentStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler"""
    # Startup
    print("🚀 Starting RAG Chatbot with Pinecone backend…")
    yield
    # Shutdown
    print("👋 Shutting down RAG Chatbot...")

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# ---------- Models ----------
class QueryRequest(BaseModel):
    question: str
    doc_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API online",
        "docs_count": len(store.docs),
        "endpoints": {
            "POST /upload": "Upload PDF",
            "GET /documents": "List documents",
            "POST /query": "Query a document",
            "DELETE /documents/{doc_id}": "Delete document",
            "GET /ui": "Web interface"
        }
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        print(f"📤 Uploading: {file.filename}")
        doc_id = store.add_document(file)
        doc_info = store.docs[doc_id]
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": doc_info["filename"],
            "chunks": doc_info["chunks"],
            "message": f"Processed {file.filename} successfully"
        }
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents():
    """List all uploaded documents"""
    return {"documents": store.list_documents()}

@app.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    """Query a specific document"""
    try:
        print(f"🔍 Query: {req.question} (doc: {req.doc_id})")
        rag = store.get_rag_pipeline(req.doc_id)
        result = rag.query(req.question)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and its index"""
    success = store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    print(f"🗑️ Deleted document: {doc_id}")
    return {"status": "deleted", "doc_id": doc_id}

# Serve static UI (if you have one)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/ui")
    async def ui():
        return FileResponse("static/index.html")
except:
    pass  # No static folder yet

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8070))
    uvicorn.run(app, host="0.0.0.0", port=port)
