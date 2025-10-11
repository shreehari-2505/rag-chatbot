from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from document_store import DocumentStore
from settings import settings
import uvicorn
import os
import io

store = DocumentStore()

# Global doc ID for demo document
DEMO_DOC_ID = "demo-document-001"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Pre-load demo document"""
    print("🚀 Starting RAG Chatbot...")
    
    # Pre-load demo document at startup
    if os.path.exists("demo_document.pdf"):
        print("📄 Pre-loading demo document...")
        try:
            # Read file content
            with open("demo_document.pdf", "rb") as f:
                file_content = f.read()
            
            # Create UploadFile object
            upload_file = UploadFile(
                filename="demo_document.pdf",
                file=io.BytesIO(file_content)
            )
            
            # Process document
            doc_id = store.add_document(upload_file)
            print(f"✅ Demo document loaded! doc_id: {doc_id}")
            
            # Store globally
            global DEMO_DOC_ID
            DEMO_DOC_ID = doc_id
        except Exception as e:
            print(f"❌ Failed to load demo document: {e}")
    
    yield
    print("👋 Shutting down...")

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# 🔥 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "message": "RAG Chatbot API - Demo Ready!",
        "demo_doc_id": DEMO_DOC_ID,
        "docs_count": len(store.docs),
        "instructions": "Use /query with doc_id to ask questions"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # 🔥 Pass UploadFile directly - document_store handles disk save
        doc_id = store.add_document(file)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "message": "Document uploaded successfully"
        }
    except Exception as e:
        print(f"Upload error: {e}")
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
        
        # 🔥 Pass doc_id to query for filtering
        result = rag.query(req.question, req.doc_id)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and its vectors"""
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
    port = int(os.environ.get("PORT", 8070))
    uvicorn.run(app, host="0.0.0.0", port=port)
