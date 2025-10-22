from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os

# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class MemoryStore:
    def __init__(self, path: str = ".chroma"):
        self.client = PersistentClient(path)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection("research_mem")

    def add(self, doc_id: str, text: str, meta: dict):
        emb = self.model.encode(text).tolist()
        self.collection.add(
            ids=[doc_id], documents=[text], metadatas=[meta], embeddings=[emb]
        )

    def search(self, query: str, k: int = 5):
        emb = self.model.encode(query).tolist()
        return self.collection.query(query_embeddings=[emb], n_results=k)

    def delete_all(self) -> None:
        """Delete everything in this collection (but keep the collection)."""
        self.collection.delete(where={})
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)
