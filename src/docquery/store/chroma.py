"""ChromaDB implementation of the VectorStore contract."""

import chromadb

from docquery.store.base import StoredChunk, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str = ".docquery_db", collection_name: str = "documents") -> None:
        client = chromadb.PersistentClient(path=persist_dir)
        self._collection = client.get_or_create_collection(collection_name)

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        self._collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int) -> list[StoredChunk]:
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        chunks: list[StoredChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        for id_, text, meta in zip(ids, docs, metas):
            chunks.append(StoredChunk(id=id_, text=text, metadata=meta))
        return chunks

    def clear(self) -> None:
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
