import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_PATH = "knowledge_base/vector_store.json"
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text):
    """Splits text into chunks by paragraph"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def ingest_document(file_path):
    """Reads a text file, chunks it, creates embeddings, and saves them to a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()

    data = {"chunks": chunks, "embeddings": embeddings}

    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"✅ Ingested {len(chunks)} chunks into the vector store.")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(query, top_k=3):
    """Given a user query, returns the most relevant chunks using cosine similarity"""
    if not os.path.exists(VECTOR_STORE_PATH):
        return []

    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    embeddings = data["embeddings"]

    query_embedding = model.encode([query])[0]

    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]

    # Get indices of top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


if __name__ == "__main__":
    ingest_document("knowledge_base/hospital_info.txt")

    test_query = "Do I need to fast before a blood test?"
    chunks = retrieve_relevant_chunks(test_query)
    print("\nDEBUG - Retrieved chunks for:", test_query)
    for c in chunks:
        print("-", c)