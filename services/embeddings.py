import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingService:
    """Service for generating sentence embeddings using HuggingFace models."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._model_loaded = False

    def _ensure_model_loaded(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        if SentenceTransformer is not None:
            try:
                # Load embedding model onto CPU cleanly on demand
                self.model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer ({str(e)}). Using basic embedding fallback.")
                self.model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of document chunks."""
        if not texts:
            return []

        self._ensure_model_loaded()
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                print(f"Error encoding documents with SentenceTransformer: {e}")

        # Deterministic fallback vector embedding generator if model fails to load
        return [self._fallback_embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate vector embedding for a single search query."""
        self._ensure_model_loaded()
        if self.model is not None:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
                return embedding.tolist()
            except Exception as e:
                print(f"Error encoding query with SentenceTransformer: {e}")

        return self._fallback_embed(text)


    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """Simple deterministic embedding fallback function for offline/test environments."""
        import hashlib
        vector = []
        text_bytes = text.encode("utf-8")
        for i in range(dim):
            h = hashlib.md5(text_bytes + str(i).encode()).hexdigest()
            # Normalize float between -1 and 1
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1
            vector.append(val)
        return vector
