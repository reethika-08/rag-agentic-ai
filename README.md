# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using an Agentic AI eBook as its knowledge source.

The chatbot retrieves relevant information from the eBook and generates answers using only the retrieved document context. If the requested information is not available in the document, the chatbot responds that it does not know the answer based on the provided document.

## Features

* PDF document ingestion
* Text chunking
* Local semantic embeddings
* FAISS vector database
* Similarity-based document retrieval
* LangGraph workflow
* Local Qwen2.5 3B language model through Ollama
* FastAPI REST API
* Retrieved context returned with every response
* Retrieval confidence score
* Protection against answering questions outside the provided document

## Architecture

```text
Agentic AI eBook PDF
        |
        v
   PDF Loader
        |
        v
 Text Chunking
        |
        v
Hugging Face Embeddings
        |
        v
   FAISS Vector DB
        |
        v
     Question
        |
        v
 LangGraph Retrieve Node
        |
        v
 Relevant Context
        |
        v
 LangGraph Generate Node
        |
        v
   Qwen2.5 3B
        |
        v
 Final Answer
 + Retrieved Context
 + Confidence Score
```

## Technologies Used

* Python 3.10
* LangChain
* LangGraph
* Hugging Face Sentence Transformers
* FAISS
* Ollama
* Qwen2.5 3B
* FastAPI
* Uvicorn
* PyPDF

## Project Structure

## Project Structure

```text
rag-agentic-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── .gitkeep
│
└── src/
    ├── __init__.py
    ├── ingestion.py
    ├── retrieve.py
    ├── rag.py
    └── graph.py

## How RAG Works in This Project

The project follows these steps:

1. The Agentic AI eBook PDF is loaded.
2. The document is divided into smaller chunks.
3. Each chunk is converted into a vector embedding.
4. The embeddings are stored in a FAISS vector database.
5. When a user asks a question, the system searches for the most relevant chunks.
6. LangGraph passes the retrieved context to the generation step.
7. Qwen2.5 3B generates an answer using the retrieved context.
8. If the question is not sufficiently related to the document, the system returns:

```text
I don't know based on the provided document.
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ollama must also be installed.

Download the local model:

```bash
ollama pull qwen2.5:3b
```

## Document Ingestion

Place the Agentic AI eBook at:

```text
data/Ebook-Agentic-AI.pdf
```

Run:

```bash
python src/ingestion.py
```

This creates the FAISS vector database inside:

```text
data/faiss_index/
```

## Running the API

Start FastAPI using:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Endpoint

### POST `/chat`

Request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Response contains:

```json
{
  "final_answer": "...",
  "retrieved_context": "...",
  "confidence_score": 0.739
}
```

## Testing

### Test 1: Document-based question

Question:

```text
What is Agentic AI according to the eBook?
```

Expected behavior:

* Relevant eBook content is retrieved.
* The answer is generated using the retrieved context.
* A retrieval confidence score is returned.

### Test 2: Out-of-document question

Question:

```text
Who won the 2022 FIFA World Cup?
```

Observed behavior:

```text
I don't know based on the provided document.
```

Confidence score:

```text
0.0
```

This demonstrates that the chatbot does not answer questions when relevant information is not available in the provided document.

### Additional Test Questions

The following questions can be used to test the chatbot:

1. What is Agentic AI according to the eBook?
2. How does Agentic AI differ from traditional AI systems?
3. What are the core components of an Agentic AI system?
4. What role does memory play in Agentic AI workflows?
5. What are multi-agent systems?
6. Who won the 2022 FIFA World Cup?

The first five questions should be answered using the eBook context, while the final question should be rejected because it is outside the document's knowledge.

## Confidence Score

The confidence score is a retrieval relevance heuristic derived from the best FAISS similarity distance.

It should be interpreted as an application-level relevance indicator, not as a calibrated probability.

Lower FAISS distance indicates greater similarity between the question and retrieved document content.

## Implementation Note

The assignment reference material suggests using OpenAI embeddings/LLM and Pinecone.

This implementation uses local alternatives:

* OpenAI embeddings → Hugging Face `all-MiniLM-L6-v2`
* OpenAI LLM → Ollama with Qwen2.5 3B
* Pinecone → FAISS

The local stack was selected so that the project can run without requiring a paid API account.

The RAG workflow and core requirements remain the same: document ingestion, embedding, retrieval, grounded generation, context output, and confidence scoring.

## Conclusion

This project demonstrates a document-grounded RAG chatbot using Python, LangGraph, FAISS, Hugging Face embeddings, Ollama, Qwen2.5 3B, and FastAPI.
