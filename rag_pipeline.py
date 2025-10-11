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
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.index = self._init_index()

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

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        print(f"📄 Extracted {len(text)} characters from PDF")
        return text

    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        print(f"✂️ Created {len(chunks)} chunks")
        return chunks

    def embed_texts(self, texts):
        print(f"🧠 Embedding {len(texts)} text(s)...")
        embeddings = list(self.embedder.embed(texts))
        return np.array(embeddings)

    def upload_to_pinecone(self, chunks, embeddings, doc_id: str):
        vectors = []
        for i, emb in enumerate(embeddings):
            vec_id = f"{doc_id}_{i}"
            vectors.append({
                "id": vec_id,
                "values": emb.tolist(),
                "metadata": {
                    "text": chunks[i],
                    "doc_id": doc_id,
                    "chunk_index": i
                }
            })
        print(f"🚀 Uploading {len(vectors)} vectors to Pinecone (doc: {doc_id})...")
        self.index.upsert(vectors=vectors)
        print("✅ Pinecone index populated!")

    def process_document(self, pdf_path, doc_id: str):
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        embeddings = self.embed_texts(chunks)
        self.upload_to_pinecone(chunks, embeddings, doc_id)
        return chunks

    def query(self, question, doc_id: str, top_k=3):
        print(f"🔍 Query: {question} (doc: {doc_id})")
        q_emb = self.embed_texts([question])[0]
        
        results = self.index.query(
            vector=q_emb.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}}
        )
        
        if not results["matches"]:
            return {
                "answer": "No relevant content found in this document.",
                "sources": []
            }
        
        contexts = [m["metadata"]["text"] for m in results["matches"]]
        prompt = self._build_prompt(question, contexts)
        answer = self._ask_groq(prompt)
        return {"answer": answer, "sources": contexts}

    def _build_prompt(self, question, contexts):
        context_str = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
        return f"""You are a helpful assistant. Answer the question based on the context below.

{context_str}

Question: {question}

Answer (be concise and cite which context you used):"""

    def _ask_groq(self, prompt):
        response = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
