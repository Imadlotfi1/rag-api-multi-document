import os
import torch
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration Section ---
# Mettre les paramètres modifiables ici pour un accès facile
DOCS_PATH = Path("Document_RAG")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
VECTOR_STORE_SEARCH_K = 3

# LangSmith Configuration (optionnel, peut être activé si besoin)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "API RAG Document Q&A"
# os.environ["LANGCHAIN_API_KEY"] = "VOTRE_CLE_API_LANGSMITH"

def load_documents_from_folder(folder_path: Path) -> List:
    """Scans a folder for PDF files and loads them into LangChain documents."""
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in {folder_path}")
        return []

    print(f"Loading {len(pdf_files)} PDF document(s)...")
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            all_docs.extend(loader.load())
            print(f"  - Loaded '{pdf_path.name}'")
        except Exception as e:
            print(f"  - Failed to load '{pdf_path.name}': {e}")
    return all_docs

def create_rag_chain():
    """Builds the complete RAG chain: Load docs, split, embed, and create retriever chain."""
    documents = load_documents_from_folder(DOCS_PATH)
    if not documents:
        raise ValueError("No documents could be loaded. RAG chain cannot be created.")

    # 1. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    # 2. Setup embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

    # 3. Create vector store from chunks
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_STORE_SEARCH_K})

    # 4. Setup LLM and prompt
    llm = OllamaLLM(model=LLM_MODEL)
    prompt_template = """Answer the user's question based only on the following context:
<context>{context}</context>
Question: {input}
Answer (in French):"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 5. Create the final retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- FastAPI Application ---
app = FastAPI(
    title="Document Q&A API",
    description="An API for querying documents using a RAG architecture.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

# Global variable to hold the RAG chain
# This is created only once when the application starts
rag_chain = None

@app.on_event("startup")
def startup_event():
    """Load models and create the RAG chain on application startup."""
    global rag_chain
    print("Application startup: Initializing RAG chain...")
    try:
        rag_chain = create_rag_chain()
        print("✅ RAG chain successfully initialized. Server is ready.")
    except Exception as e:
        print(f"🔥 FATAL ERROR during startup: {e}")
        # In a real-world scenario, you might want the app to fail if the chain can't be built.
        # For now, we'll let it run but the endpoint will fail.
        rag_chain = None

@app.post("/ask", response_model=QueryResponse, summary="Ask a question to the document base")
def ask_question(request: QueryRequest):
    """Receives a question and returns an answer based on the loaded documents."""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain is not available due to a startup error.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = rag_chain.invoke({"input": request.question})
        return QueryResponse(question=request.question, answer=response["answer"])
    except Exception as e:
        # Generic error handling for any unexpected issues during invocation
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the question: {e}")
