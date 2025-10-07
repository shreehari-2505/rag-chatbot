import uuid, shutil, json
from pathlib import Path
from pinecone import Pinecone
from rag_pipeline import RAGPipeline
from settings import settings

class DocumentStore:
    """Manages uploaded PDFs and their Pinecone indexes."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.upload_dir = Path(settings.uploads_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Persist document mappings
        self.docs_file = self.upload_dir / "documents.json"
        self.docs = self._load_docs()
    
    def _load_docs(self):
        if self.docs_file.exists():
            with open(self.docs_file) as f:
                return json.load(f)
        return {}
    
    def _save_docs(self):
        with open(self.docs_file, 'w') as f:
            json.dump(self.docs, f, indent=2)
    
    def _make_index_name(self, doc_id: str) -> str:
        return f"{settings.index_prefix}{doc_id}"
    
    def list_documents(self):
        return [
            {
                "doc_id": doc_id,
                "filename": d["filename"],
                "index_name": d["index_name"],
                "chunks": d["chunks"]
            }
            for doc_id, d in self.docs.items()
        ]
    
    def add_document(self, upload_file):
        file_path = self.upload_dir / upload_file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)
        
        doc_id = str(uuid.uuid4())
        index_name = self._make_index_name(doc_id)
        
        rag = RAGPipeline(
            groq_api_key=settings.groq_api_key,
            pinecone_client=self.pc,
            index_name=index_name,
        )
        
        chunks = rag.process_document(str(file_path))
        self.docs[doc_id] = {
            "filename": upload_file.filename,
            "index_name": index_name,
            "chunks": len(chunks)
        }
        self._save_docs()  # Persist to disk
        return doc_id
    
    def get_rag_pipeline(self, doc_id: str) -> RAGPipeline:
        if doc_id not in self.docs:
            raise KeyError("Document not found")
        info = self.docs[doc_id]
        return RAGPipeline(settings.groq_api_key, self.pc, info["index_name"])
    
    def delete_document(self, doc_id: str):
        if doc_id not in self.docs:
            return False
        index_name = self.docs[doc_id]["index_name"]
        self.pc.delete_index(index_name)
        self.docs.pop(doc_id)
        self._save_docs()  # Persist to disk
        return True
