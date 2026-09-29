from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_PATH = "data/faiss_index"


def main():
    print("Loading FAISS database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    question = input("\nEnter your question: ")

    results = vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    print("\n--- Retrieved Context ---")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {score}")
        print(document.page_content[:1000])
        print("-" * 60)


if __name__ == "__main__":
    main()