from jev_rag_research.shared.embeddings import embed_text
from jev_rag_research.shared.retrieval import retrieve


documents = [
    "Paris is the capital city of France.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Photosynthesis allows plants to convert light energy into chemical energy.",
]


def main():
    print("Creating document embeddings...")

    embeddings = [
        embed_text(document)
        for document in documents
    ]

    query = "What is the capital of France?"

    print(f"\nQuery: {query}")

    query_embedding = embed_text(query)

    results = retrieve(
        query_embedding,
        embeddings,
        top_k=2,
    )

    print("\nTop results:")

    for index, score in results:
        print(f"\nScore: {score:.4f}")
        print(f"Document: {documents[index]}")


if __name__ == "__main__":
    main()