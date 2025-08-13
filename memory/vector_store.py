from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class MemoryStore:
    def __init__(self, path: str = ".chroma"):
        self.client = PersistentClient(path)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection("research_mem")

    def add(self, doc_id: str, text: str, meta: dict):
        emb = self.model.encode(text).tolist()
        self.collection.add(ids=[doc_id], documents=[text], metadatas=[meta], embeddings=[emb])

    def search(self, query: str, k: int = 5):
        emb = self.model.encode(query).tolist()
        return self.collection.query(query_embeddings=[emb], n_results=k)
