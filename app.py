from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import app as rag_app


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot using LangGraph, FAISS, Hugging Face embeddings and Ollama.",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    result = rag_app.invoke({
        "question": request.query,
        "context": "",
        "answer": "",
        "confidence_score": 0.0
    })

    retrieved_context_chunks = (
        result["context"].split("\n\n")
        if result["context"]
        else []
    )

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": retrieved_context_chunks,
        "confidence_score": result["confidence_score"]
    }