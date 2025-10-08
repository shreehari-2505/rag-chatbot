import PyPDF2
import numpy as np
import uuid
from groq import Groq
from fastembed import TextEmbedding
from pinecone import ServerlessSpec

class RAGPipeline:
    def __init__(self, groq_api_key, pinecone_client, index_name: str):
        self.groq = Groq(api_key=groq_api_key)
        self.pc = pinecone_client
        self.index_name = index_name
        
        # Initialize FastEmbed (lightweight, CPU-optimized!)
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        self.index = self._init_index()

    # ---- Pinecone setup ----
    def _init_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"📦 Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        return self.pc.Index(self.index_name)

    # ---- PDF extraction ---- 🔥 FIXED: Uses file_path from disk
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file on disk"""
        text = ""
        try:
            # 🚀 Open the saved file from disk (NOT upload_file.file)
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                        
            print(f"✅ Extracted {len(text)} characters from {pdf_path}")
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")
        return text

    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    # ---- Embeddings + Pinecone ----
    def embed_texts(self, texts):
        """Fast embeddings with ONNX runtime"""
        print(f"🧠 Embedding {len(texts)} texts with FastEmbed...")
        
        # FastEmbed returns generator, convert to list
        embeddings = list(self.embedder.embed(texts))
        return np.array(embeddings, dtype='float32')

    def upload_to_pinecone(self, chunks, embeddings):
        vectors = []
        for i, emb in enumerate(embeddings):
            vec_id = str(uuid.uuid4())
            vectors.append({
                "id": vec_id,
                "values": emb.tolist(),
                "metadata": {"text": chunks[i]}
            })
        print(f"🚀 Uploading {len(vectors)} vectors to Pinecone...")
        self.index.upsert(vectors=vectors)
        print("✅ Pinecone index populated!")

    def process_document(self, pdf_path):
        """Process PDF from disk path"""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        embeddings = self.embed_texts(chunks)
        self.upload_to_pinecone(chunks, embeddings)
        return chunks

    # ---- Querying ----
    def query(self, question, top_k=3):
        print(f"🔍 Query: {question}")
        q_emb = self.embed_texts([question])[0]
        results = self.index.query(
            vector=q_emb.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        contexts = [m["metadata"]["text"] for m in results["matches"]]
        prompt = self._build_prompt(question, contexts)
        answer = self._ask_groq(prompt)
        return {"answer": answer, "sources": contexts}

    def _build_prompt(self, question, contexts):
        context_str = "".join(
            f"[Context {i+1}]: {c}" for i, c in enumerate(contexts)
        )
        return f"""You are a helpful AI assistant. Answer the question based on the context below. Context: {context_str} Question: {question} Answer naturally and conversationally. If the info isn't in the documents, just say "I don't see that information in your documents." Keep it friendly and clear!"""

    def _ask_groq(self, prompt):
        response = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "system", 
                "content": "You're a friendly document assistant. Answer questions based on the provided context naturally and conversationally. If info isn't in the documents, say so honestly."
            },{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()
