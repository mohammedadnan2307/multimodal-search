"""Vector storage for embeddings."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

import config


@dataclass
class IndexItem:
    id: str
    modality: str  # "text" or "image"
    document: str
    page: int
    content: Optional[str] = None


class VectorStore:
    
    def __init__(self, dim: int = None):
        self.dim = dim or config.EMBEDDING_DIM
        self.embeddings: list[np.ndarray] = []
        self.items: list[IndexItem] = []
        self.embedder_type = "unknown"
    
    def add(self, embedding: np.ndarray, item: IndexItem):
        """Add an item to the store."""
        self.embeddings.append(embedding)
        self.items.append(item)
    
    def add_batch(self, embeddings: np.ndarray, items: list[IndexItem]):
        """Add multiple items at once."""
        for emb, item in zip(embeddings, items):
            self.add(emb, item)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        modality: str = None,
        document: str = None
    ) -> list[tuple[IndexItem, float]]:
        """Search for similar items."""
        if not self.embeddings:
            return []
        
        top_k = top_k or config.DEFAULT_TOP_K
        
        # Build embedding matrix
        matrix = np.vstack(self.embeddings)
        
        # Compute cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = matrix @ query_norm
        
        # Filter and rank
        results = []
        for idx, score in enumerate(similarities):
            item = self.items[idx]
            
            # Apply filters
            if modality and item.modality != modality:
                continue
            if document and item.document != document:
                continue
            if score < config.SIMILARITY_THRESHOLD:
                continue
            
            results.append((item, float(score)))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def save(self, path: Path = None):
        """Save store to disk."""
        path = path or config.INDEX_DIR
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.embeddings:
            matrix = np.vstack(self.embeddings)
            np.save(path / "embeddings.npy", matrix)
        
        # Save metadata
        metadata = {
            "dim": self.dim,
            "embedder_type": self.embedder_type,
            "items": [asdict(item) for item in self.items]
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        print(f"Saved {len(self.items)} items to {path}")
    
    @classmethod
    def load(cls, path: Path = None) -> "VectorStore":
        """Load store from disk."""
        path = path or config.INDEX_DIR
        
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        store = cls(dim=metadata["dim"])
        store.embedder_type = metadata.get("embedder_type", "unknown")
        
        # Load items
        for item_dict in metadata["items"]:
            store.items.append(IndexItem(**item_dict))
        
        # Load embeddings
        embeddings = np.load(path / "embeddings.npy")
        store.embeddings = [embeddings[i] for i in range(len(embeddings))]
        
        print(f"Loaded {len(store.items)} items from {path}")
        return store
    
    def stats(self) -> dict:
        """Get store statistics."""
        text_count = sum(1 for item in self.items if item.modality == "text")
        image_count = sum(1 for item in self.items if item.modality == "image")
        documents = set(item.document for item in self.items)
        
        return {
            "total_items": len(self.items),
            "text_items": text_count,
            "image_items": image_count,
            "documents": len(documents),
            "embedder": self.embedder_type
        }
