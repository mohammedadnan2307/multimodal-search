"""Search engine combining embeddings and storage."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.embeddings import Embedder, get_embedder
from src.storage import VectorStore, IndexItem

import config


class SearchMode(Enum):
    ALL = "all"
    TEXT = "text"
    IMAGE = "image"


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    modality: str
    document: str
    page: int
    content: str


class SearchEngine:
    """Multimodal search engine."""
    
    def __init__(self, store: VectorStore = None, embedder: Embedder = None):
        self.store = store
        self.embedder = embedder
        
        # Auto-load if store exists
        if store is None:
            try:
                self.store = VectorStore.load()
                # Match embedder to stored type
                use_clip = self.store.embedder_type == "clip"
                self.embedder = get_embedder(use_clip=use_clip)
            except FileNotFoundError:
                pass
    
    def search(
        self,
        query: str,
        top_k: int = None,
        mode: SearchMode = SearchMode.ALL,
        document: str = None
    ) -> list[SearchResult]:
        """Search for relevant content."""
        if not self.store or not self.embedder:
            return []
        
        top_k = top_k or config.DEFAULT_TOP_K
        
        # Embed query
        query_embedding = self.embedder.embed_text([query])[0]
        
        # Determine modality filter
        modality = None
        if mode == SearchMode.TEXT:
            modality = "text"
        elif mode == SearchMode.IMAGE:
            modality = "image"
        
        # Search
        results = self.store.search(
            query_embedding,
            top_k=top_k,
            modality=modality,
            document=document
        )
        
        # Format results
        return [
            SearchResult(
                id=item.id,
                score=score,
                modality=item.modality,
                document=item.document,
                page=item.page,
                content=item.content or ""
            )
            for item, score in results
        ]
    
    def index_documents(self, use_clip: bool = True) -> dict:
        """Index all documents in the PDF directory."""
        from src.ingestion import process_pdfs
        
        # Process PDFs
        chunks, images = process_pdfs()
        
        if not chunks and not images:
            return {"error": "No content found"}
        
        # Initialize embedder and store
        self.embedder = get_embedder(use_clip=use_clip)
        self.store = VectorStore(dim=self.embedder.dim)
        self.store.embedder_type = "clip" if use_clip else "demo"
        
        # Embed and store text chunks
        if chunks:
            print(f"Embedding {len(chunks)} text chunks...")
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_text(texts)
            
            items = [
                IndexItem(
                    id=c.chunk_id,
                    modality="text",
                    document=c.document,
                    page=c.page,
                    content=c.content
                )
                for c in chunks
            ]
            self.store.add_batch(embeddings, items)
        
        # Embed and store images
        if images:
            print(f"Embedding {len(images)} images...")
            paths = [img.path for img in images]
            embeddings = self.embedder.embed_images(paths)
            
            items = [
                IndexItem(
                    id=img.image_id,
                    modality="image",
                    document=img.document,
                    page=img.page,
                    content=str(img.path)
                )
                for img in images
            ]
            self.store.add_batch(embeddings, items)
        
        self.store.save()
        
        return self.store.stats()
