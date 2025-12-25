"""Embedding models for text and images."""

import hashlib
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image

import config


class Embedder(ABC):
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        pass
    
    @abstractmethod
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Embed a list of images."""
        pass


class CLIPEmbedder(Embedder):
    """CLIP-based embedder for text and images."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.CLIP_MODEL
        self.device = device or config.DEVICE
        self._model = None
        self._processor = None
        self._dim = None
    
    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim
    
    def _load_model(self):
        """Load CLIP model on first use."""
        if self._model is not None:
            return
            
        import torch
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPImageProcessor
        from huggingface_hub import snapshot_download
        
        print(f"Loading CLIP model: {self.model_name}")
        
        # Download to local directory to avoid Windows symlink issues
        model_dir = config.DATA_DIR / "clip_model"
        model_dir.mkdir(exist_ok=True)
        
        snapshot_download(
            repo_id=self.model_name,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        
        # Load components
        tokenizer = CLIPTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        image_processor = CLIPImageProcessor(
            size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224},
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        
        self._processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
        self._model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
        self._model = self._model.to(self.device).eval()
        self._dim = self._model.config.projection_dim
        
        print(f"CLIP ready (dim={self._dim}, device={self.device})")
    
    def embed_text(self, texts: list[str]) -> np.ndarray:
        import torch
        self._load_model()
        
        embeddings = []
        for i in range(0, len(texts), config.BATCH_SIZE):
            batch = texts[i:i + config.BATCH_SIZE]
            inputs = self._processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                embeddings.append(outputs.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        import torch
        self._load_model()
        
        embeddings = []
        for i in range(0, len(image_paths), config.BATCH_SIZE):
            batch_paths = image_paths[i:i + config.BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                embeddings.append(outputs.cpu().numpy())
        
        return np.vstack(embeddings)


class DemoEmbedder(Embedder):
    """Hash-based embedder for testing without GPU."""
    
    def __init__(self, dim: int = None):
        self._dim = dim or config.EMBEDDING_DIM
        
        # Common terms for semantic similarity
        self._vocab = [
            "exercise", "workout", "training", "muscle", "strength", "body",
            "push", "pull", "squat", "plank", "core", "arm", "leg", "back",
            "fitness", "health", "technique", "position", "movement", "form"
        ]
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def _hash_embed(self, text: str) -> np.ndarray:
        """Create embedding from text hash + semantic features."""
        text_lower = text.lower()
        
        # Base embedding from hash
        hash_bytes = hashlib.sha256(text_lower.encode()).digest()
        base = np.frombuffer(hash_bytes, dtype=np.uint8)
        base = np.tile(base, self._dim // len(base) + 1)[:self._dim]
        embedding = (base.astype(np.float32) - 128) / 128
        
        # Add semantic signal from vocabulary matches
        for i, word in enumerate(self._vocab):
            if word in text_lower:
                idx = (i * 17) % self._dim
                embedding[idx:idx + 10] += 0.5
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def embed_text(self, texts: list[str]) -> np.ndarray:
        return np.array([self._hash_embed(t) for t in texts])
    
    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        # For demo, hash the filename
        return np.array([self._hash_embed(str(p.name)) for p in image_paths])


def get_embedder(use_clip: bool = True) -> Embedder:
    """Get the appropriate embedder."""
    if use_clip:
        try:
            import torch
            return CLIPEmbedder()
        except ImportError:
            print("PyTorch not available, using demo embedder")
    return DemoEmbedder()
