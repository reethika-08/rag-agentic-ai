from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


VECTORSTORE_PATH = "data/faiss_index"


def load_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    return vectorstore, llm


def ask_question(question):
    vectorstore, llm = load_system()

    results = vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    context = "\n\n".join(
        document.page_content for document, score in results
    )

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Do not use outside knowledge.
Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content, results


if __name__ == "__main__":
    question = input("Enter your question: ")

    answer, results = ask_question(question)

    print("\n--- ANSWER ---")
    print(answer)

    print("\n--- RETRIEVED CONTEXT ---")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\nResult {i} | Distance: {score}")
        print(document.page_content[:1000])