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

The chatbot was tested with both in-document and out-of-document questions to verify retrieval, grounded generation, and refusal behavior.

### Test 1 — Agentic AI Definition

**Question:**

> What is Agentic AI according to the eBook?

**Result:**

* Answer generated from the retrieved eBook context.
* Confidence score: `0.739`
* Status: **Passed**

### Test 2 — Core Components

**Question:**

> What are the core components of an Agentic AI system?

**Result:**

* Correctly identified Perception, Reasoning, Planning, Learning, and Execution.
* Confidence score: `0.665`
* Status: **Passed**

### Test 3 — Multi-Agent Systems

**Question:**

> What are multi-agent systems?

**Result:**

* Retrieved the relevant Multi-Agent Systems section and generated a grounded answer.
* Confidence score: `0.716`
* Status: **Passed**

### Test 4 — Out-of-Document Question

**Question:**

> Who won the 2022 FIFA World Cup?

**Result:**

* The chatbot refused to answer because the information was not available in the provided document.
* Answer: `I don't know based on the provided document.`
* Confidence score: `0.000`
* Status: **Passed**

### Test 5 — Agentic AI vs Traditional AI

**Question:**

> How does Agentic AI differ from traditional AI systems?

**Result:**

* Correctly explained the reactive vs. proactive distinction using retrieved eBook context.
* Confidence score: `0.644`
* Status: **Passed**

### Test Summary

| Test                         | Type            | Status |
| ---------------------------- | --------------- | ------ |
| Agentic AI definition        | In-document     | Passed |
| Core components              | In-document     | Passed |
| Multi-Agent Systems          | In-document     | Passed |
| FIFA World Cup question      | Out-of-document | Passed |
| Agentic AI vs traditional AI | In-document     | Passed |

These tests demonstrate that the system can retrieve relevant document content, generate grounded answers, provide a relevance-based confidence score, and refuse questions that fall outside the available knowledge source.


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
