import uuid
import shutil
import json
from pathlib import Path
from pinecone import Pinecone
from settings import settings

class DocumentStore:
    """Manages uploaded PDFs and their Pinecone indexes."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.upload_dir = Path(settings.uploads_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # 🔥 ONE SHARED INDEX FOR ALL DOCUMENTS
        self.shared_index_name = "rag-chatbot-shared"
        self._ensure_shared_index()
        
        # Persist document mappings
        self.docs_file = self.upload_dir / "documents.json"
        self.docs = self._load_docs()
    
    def _ensure_shared_index(self):
        """Create shared index if it doesn't exist"""
        from pinecone import ServerlessSpec
        existing = [i.name for i in self.pc.list_indexes()]
        if self.shared_index_name not in existing:
            print(f"📦 Creating shared Pinecone index: {self.shared_index_name}")
            self.pc.create_index(
                name=self.shared_index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print("✅ Shared index created!")
    
    def _load_docs(self):
        if self.docs_file.exists():
            with open(self.docs_file) as f:
                return json.load(f)
        return {}
    
    def _save_docs(self):
        with open(self.docs_file, 'w') as f:
            json.dump(self.docs, f, indent=2)
    
    def list_documents(self):
        return [
            {
                "doc_id": doc_id,
                "filename": d["filename"],
                "chunks": d["chunks"]
            }
            for doc_id, d in self.docs.items()
        ]
    
    def add_document(self, upload_file):
        """Handle uploaded file and add to document store"""
        # Create unique filename
        filename = upload_file.filename
        file_path = self.upload_dir / filename
        
        # 🔥 Save file to disk ONCE
        with open(file_path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)
        print(f"📄 Saved file to: {file_path}")
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # 🔥 Create RAG pipeline with SHARED index
        rag = RAGPipeline(
            groq_api_key=settings.groq_api_key,
            pinecone_client=self.pc,
            index_name=self.shared_index_name,
        )
        
        # 🔥 Pass doc_id to process_document
        chunks = rag.process_document(str(file_path), doc_id)
        
        # Store document metadata
        self.docs[doc_id] = {
            "filename": filename,
            "chunks": len(chunks)
        }
        self._save_docs()
        
        print(f"✅ Document added: {doc_id} ({len(chunks)} chunks)")
        return doc_id
    
    def get_rag_pipeline(self, doc_id: str) -> RAGPipeline:
        """Get RAG pipeline for querying specific document"""
        if doc_id not in self.docs:
            raise KeyError("Document not found")
        
        # 🔥 Return pipeline with shared index
        return RAGPipeline(
            settings.groq_api_key, 
            self.pc, 
            self.shared_index_name
        )
    
    def delete_document(self, doc_id: str):
        """Delete document vectors from shared index"""
        if doc_id not in self.docs:
            return False
        
        # 🔥 Delete only this doc's vectors using metadata filter
        rag = RAGPipeline(settings.groq_api_key, self.pc, self.shared_index_name)
        
        # Delete vectors by doc_id
        rag.index.delete(filter={"doc_id": {"$eq": doc_id}})
        
        # Remove from docs tracking
        filename = self.docs[doc_id]["filename"]
        file_path = self.upload_dir / filename
        if file_path.exists():
            file_path.unlink()  # Delete file
        
        self.docs.pop(doc_id)
        self._save_docs()
        
        print(f"🗑️ Deleted document: {doc_id}")
        return True
