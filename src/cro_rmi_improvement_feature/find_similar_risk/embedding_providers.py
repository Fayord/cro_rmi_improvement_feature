from typing import List, Dict
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json
import hashlib
import os
from pathlib import Path
from google import genai
import time


class BaseEmbeddingProvider:
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, text: str) -> Path:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if self.model_name is None:
            # raise
            raise Exception("self.model_name is None")
        return (
            self.cache_dir
            / f"{self.__class__.__name__}_{self.model_name}_{text_hash}.json"
        )

    def get_embedding(self, text: str) -> np.ndarray:
        cache_path = self._get_cache_path(text)
        try:
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    return np.array(json.load(f))
        except json.JSONDecodeError:
            # Handle the case where the file is not valid JSON
            print(f"Invalid JSON file: {cache_path}")
            # remove that cache file
            os.remove(cache_path)
        embedding = self._get_embedding_impl(text)

        with open(cache_path, "w") as f:
            json.dump(embedding.tolist(), f)

        return embedding

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.client = OpenAI()

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return np.array(response.data[0].embedding)


# embedding_providers for gemini google
class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "gemini-embedding-exp-03-07",
        api_key: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY must be provided either as an argument or environment variable"
            )
        self.client = genai.Client(api_key=api_key)

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        max_retries = 5
        base_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                return np.array(result.embeddings[0].values)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise  # Re-raise the exception if it's not a rate limit error or we're out of retries

        raise Exception(f"Failed to get embedding after {max_retries} attempts")


class SentenceTransformerProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]
