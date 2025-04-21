from typing import List, Dict
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json
import hashlib
import os
from pathlib import Path


class BaseEmbeddingProvider:
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, text: str) -> Path:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{self.__class__.__name__}_{text_hash}.json"

    def get_embedding(self, text: str) -> np.ndarray:
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return np.array(json.load(f))

        embedding = self._get_embedding_impl(text)

        with open(cache_path, "w") as f:
            json.dump(embedding.tolist(), f)

        return embedding

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = OpenAI()
    
    def _get_embedding_impl(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model = SentenceTransformer(model_name)

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]
