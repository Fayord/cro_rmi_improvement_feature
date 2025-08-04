import random
import time
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from openai import OpenAI, RateLimitError
from google import genai
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging to output to console
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEmbeddingProvider(ABC):
    def __init__(self, cache_dir: str = ".embedding_cache", **kwargs):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        cache_key = self._generate_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")

        if use_cache and os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(
                    f"Error loading from cache: {e}. Re-generating embedding."
                )

        embedding = self._get_embedding_impl(text)
        if embedding is not None:
            if use_cache:
                np.save(cache_file, embedding)
            return embedding
        return np.array([])

    @abstractmethod
    def _get_embedding_impl(self, text: str) -> np.ndarray:
        pass

    def _generate_cache_key(self, text: str) -> str:
        import hashlib

        return hashlib.md5(text.encode()).hexdigest()


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY must be provided either as an argument or environment variable"
            )
        self.client = OpenAI(api_key=api_key)

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        max_retries = 5
        base_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model_name,
                )
                return np.array(response.data[0].embedding)
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + (
                        random.random() * 0.1
                    )  # Exponential backoff with jitter
                    logger.warning(
                        f"Rate limit exceeded for OpenAI. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to get embedding after {max_retries} attempts due to rate limit: {e}"
                    )
                    raise  # Re-raise the exception if out of retries
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while getting embedding: {e}"
                )
                raise


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
        self.model = SentenceTransformer(model_name)

    def _get_embedding_impl(self, text: str) -> np.ndarray:
        return self.model.encode(text)
