from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

VECTORSTORE_PATH = "data/faiss_index"


class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    confidence_score: float


def retrieve(state: AgentState):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search_with_score(
        state["question"],
        k=3
    )

    distances = [float(score) for _, score in results]
    best_distance = min(distances)

    if best_distance > 0.65:
        context = ""
        confidence = 0.0
    else:
        context = "\n\n".join(
            document.page_content for document, score in results
        )
        confidence = float(1 / (1 + best_distance))

    return {
        "context": context,
        "confidence_score": round(confidence, 3)
    }


def generate(state: AgentState):
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    if not state["context"]:
        return {
            "answer": "I don't know based on the provided document."
        }

    prompt = f"""
You are a strict document-based AI assistant.

Answer the user's question ONLY using the CONTEXT below.

Rules:
1. Use only information explicitly supported by the context.
2. Do not use your own knowledge.
3. Do not add facts that are not present in the context.
4. If the context does not contain enough information to answer the question, say:
"I don't know based on the provided document."
5. Do not guess or infer missing facts.
6. Keep the answer concise.

CONTEXT:
{state["context"]}

QUESTION:
{state["question"]}

ANSWER:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


if __name__ == "__main__":
    question = input("Enter your question: ")

    result = app.invoke({
        "question": question,
        "context": "",
        "answer": "",
        "confidence_score": 0.0
    })

    print("\n--- FINAL ANSWER ---")
    print(result["answer"])

    print("\n--- CONFIDENCE SCORE ---")
    print(result["confidence_score"])

    print("\n--- RETRIEVED CONTEXT ---")
    print(result["context"])