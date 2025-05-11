# === IMPORTS ===
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# LangChain & Vector DB stuff
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# RAG & LLM stuff
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain import hub

# === FASTAPI APP SETUP ===
app = FastAPI()

# CORS for local dev (open to all, lock down for prod!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === VECTOR STORE DIRECTORY ===
VECTOR_DB_FOLDER = "chroma_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# === EMBEDDINGS & LLM CONFIG ===
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_API_KEY = "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"
LLM_MODEL = "qwen/qwen-plus"
LLM_BASE_URL = "https://openrouter.ai/api/v1"

# === ENDPOINT 1: UPLOAD PDF ===
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Save PDF to disk
    2. Split PDF into chunks
    3. Embed chunks & store in vector DB
    4. Delete temp PDF
    """
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as temp_file:
        temp_file.write(await file.read())
    try:
        # Load and split PDF
        pdf_loader = PyPDFLoader(temp_pdf_path)
        pdf_pages = pdf_loader.load()
        chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        page_chunks = chunker.split_documents(pdf_pages)

        # Create embeddings and vector DB
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = Chroma.from_documents(page_chunks, embedder, persist_directory=VECTOR_DB_FOLDER)
        vector_db.persist()
    finally:
        os.remove(temp_pdf_path)

    return {"message": f"Uploaded and indexed {file.filename}!"}

# === ENDPOINT 2: ASK QUESTION (RAG) ===
class QuestionRequest(BaseModel):
    query: str

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    """
    1. Retrieve relevant chunks using retriever
    2. Use RAG chain (retriever + LLM) to answer
    """
    # Load embeddings and vector DB
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embedder)

    # Build retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Build LLM
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL
    )

    # Build RAG chain
    rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Run the chain
    response = retrieval_chain.invoke({"input": request.query})
    answer = response.get("answer", "No answer found.")

    return {"answer": answer}

# === RUN LOCALLY ===
# Save this as main.py, then run:
#   uvicorn main:app --reload
